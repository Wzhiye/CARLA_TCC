"""Pygame operator view: front camera of the selected robotaxi with rear-view mirrors,
fleet HUD and an action bar.

Keyboard (all actions also work by clicking the buttons at the bottom):

    1/2/3        select robotaxi                 TAB   next open RA request
    LEFT/RIGHT   bypass left / right             Q/E   turn left / right at next junction
    P            pull over                       R     resume
    X            abort the running maneuver
    ENTER        confirm a pending command (or press/click the same action again)
    T            inject the chosen scenario on the selected robotaxi
    Y            choose which scenario T injects    F1..F5  inject S1..S5 directly
    C            clear the incident blocking the selected robotaxi (Shift+C: all)
    V            camera: front/chase/top         M     rear-view mirrors on/off
    F            server window camera follows the selection
    D            debug drawing: subtle/full/off  H     help
    ESC          cancel a pending command, otherwise quit

Grey buttons are actions the selected robotaxi would reject right now; hover
over one to see why. Letter keys need an English keyboard layout: an active
Chinese IME swallows them, so text input is switched off for this window
(and the buttons always work).
"""

import logging
import threading
import time

import carla
import numpy as np
import pygame

from ra_demo.core import bus as ra

LOG = logging.getLogger(__name__)

CAMERA_MODES = ("front", "chase", "top")


def camera_transform(mode, vehicle):
    """Main camera placement for ``mode``, scaled to the vehicle's size."""
    extent = vehicle.bounding_box.extent
    length, height = 0.5 + extent.x, 0.5 + extent.z
    if mode == "front":
        # Just above the windshield base, looking ahead (CARLA manual_control's hood view).
        return carla.Transform(carla.Location(x=0.8 * length, z=1.3 * height))
    if mode == "chase":
        return carla.Transform(carla.Location(x=-2.6 * length, z=2.2 * height), carla.Rotation(pitch=-10.0))
    return carla.Transform(carla.Location(x=6.0, z=38.0), carla.Rotation(pitch=-90.0))


# Rear-view mirrors: (key, placement on the car, field of view, image size).
# The images are flipped left-right when drawn, like a real mirror.
def mirror_specs(vehicle):
    extent = vehicle.bounding_box.extent
    side = extent.y + 0.15
    return (
        ("rear", carla.Transform(carla.Location(x=-0.9 * extent.x, z=1.3 * (0.5 + extent.z)),
                                 carla.Rotation(yaw=180.0, pitch=-4.0)), 60, (480, 132)),
        ("left", carla.Transform(carla.Location(x=0.6, y=-side, z=1.05),
                                 carla.Rotation(yaw=-160.0, pitch=-6.0)), 55, (288, 162)),
        ("right", carla.Transform(carla.Location(x=0.6, y=side, z=1.05),
                                  carla.Rotation(yaw=160.0, pitch=-6.0)), 55, (288, 162)),
    )

STATE_COLORS = {
    "IDLE": (180, 180, 180),
    "CRUISING": (120, 220, 120),
    "DWELLING": (120, 180, 255),
    "BLOCKED": (255, 200, 80),
    "AWAITING_RA": (255, 90, 90),
    "EXECUTING": (110, 200, 255),
    "PULLED_OVER": (210, 160, 255),
}

# (id, label, key, needs confirmation)
BUTTONS = (
    (ra.ACTION_BYPASS_LEFT, "<- Bypass L", pygame.K_LEFT, True),
    (ra.ACTION_BYPASS_RIGHT, "Bypass R ->", pygame.K_RIGHT, True),
    (ra.ACTION_TURN_LEFT, "Q Turn L", pygame.K_q, True),
    (ra.ACTION_TURN_RIGHT, "E Turn R", pygame.K_e, True),
    (ra.ACTION_PULL_OVER, "P Pull over", pygame.K_p, True),
    (ra.ACTION_RESUME, "R Resume", pygame.K_r, True),
    (ra.ACTION_ABORT, "X Abort", pygame.K_x, False),
    ("INJECT", "T Inject", pygame.K_t, False),
    ("SCENARIO", "Y Next scen.", pygame.K_y, False),
    ("CLEAR", "C Clear", pygame.K_c, False),
)
CONTROL_BUTTONS = ("INJECT", "SCENARIO", "CLEAR")
FUNCTION_KEYS = (pygame.K_F1, pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F5,
                 pygame.K_F6, pygame.K_F7, pygame.K_F8)
KEY_TO_BUTTON = {key: button_id for button_id, _, key, _ in BUTTONS}
NEEDS_CONFIRM = {button_id: confirm for button_id, _, _, confirm in BUTTONS}

HELP_LINES = [
    "1/2/3 select   TAB next request   V camera   M mirrors   F follow   D debug draw   H help",
    "Actions: <-/-> bypass   Q/E turn   P pull over   R resume   X abort   (ENTER confirms)",
    "T inject chosen scenario on selected (Y: next scenario, F1..F5 direct)   C clear its incident (Shift+C all)",
    "Grey buttons: the car would reject them - hover to see why.   ESC cancel/quit",
]

BAR_HEIGHT = 44
CONFIRM_TIMEOUT_S = 6.0
PANEL_WIDTH = 430                 # fleet panel (left) and request panel (right)
REAR_MIRROR = (400, 110)          # drawn size of the rear-view mirror (top centre)
SIDE_MIRROR = (240, 135)          # drawn size of each side mirror (bottom corners)


class Viewer:
    """Single-window pygame view with camera switching and operator actions."""

    def __init__(self, sim, fleet, scenarios, bus, drawer, cfg):
        self.sim = sim
        self.fleet = fleet
        self.scenarios = scenarios
        self.bus = bus
        self.drawer = drawer
        viewer_cfg = cfg.get("viewer", {})
        self.width = int(viewer_cfg.get("width", 1280))
        self.height = int(viewer_cfg.get("height", 720))
        self.fov = float(viewer_cfg.get("fov", 90.0))
        self.follow = bool(viewer_cfg.get("spectator_follow", True))
        self.confirm = bool(viewer_cfg.get("confirm_commands", True))
        self.mirrors = bool(viewer_cfg.get("mirrors", True))
        self.mirror_tick = 1.0 / max(1.0, float(viewer_cfg.get("mirror_fps", 10.0)))
        self.show_help = False
        self.selected = 0
        wanted = str(viewer_cfg.get("camera", "front"))
        self.mode = CAMERA_MODES.index(wanted) if wanted in CAMERA_MODES else 0
        self._camera = None
        self._mirror_cameras = {}
        self._frame = None
        self._mirror_frames = {}
        self._lock = threading.Lock()
        self._display = None
        self._font = None
        self._font_big = None
        self._last_wall = time.perf_counter()
        self._fps = 0.0
        self._pending = None          # (action, vehicle display name, wall time)
        self._button_rects = {}
        self._snapshot = None
        self._scenario_index = 0

    # ----------------------------------------------------------------- setup
    def init(self):
        pygame.init()
        pygame.display.set_caption("Robotaxi RA Demo")
        self._display = pygame.display.set_mode((self.width, self.height),
                                                pygame.HWSURFACE | pygame.DOUBLEBUF)
        # An active IME (e.g. Chinese Pinyin) turns letter keys into text composition
        # and pygame never sees them as key presses. Switch text input off.
        pygame.key.stop_text_input()
        self._font = pygame.font.SysFont("consolas,couriernew,dejavusansmono,monospace", 15)
        self._font_big = pygame.font.SysFont("consolas,couriernew,dejavusansmono,monospace", 20,
                                             bold=True)
        self._attach_camera()

    def _spawn_camera(self, vehicle, transform, size, fov, tick=None):
        blueprint = self.sim.world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(size[0]))
        blueprint.set_attribute("image_size_y", str(size[1]))
        blueprint.set_attribute("fov", str(fov))
        if tick is not None and blueprint.has_attribute("sensor_tick"):
            blueprint.set_attribute("sensor_tick", "{:.3f}".format(tick))
        return self.sim.world.spawn_actor(blueprint, transform, attach_to=vehicle,
                                          attachment_type=carla.AttachmentType.Rigid)

    def _attach_camera(self):
        self._destroy_camera()
        if not self.fleet.robotaxis:
            return
        vehicle = self.fleet.robotaxis[self.selected].vehicle
        self._camera = self._spawn_camera(vehicle, camera_transform(CAMERA_MODES[self.mode], vehicle),
                                          (self.width, self.height), self.fov)
        self._camera.listen(lambda image: self._on_image(image, None))
        if self.mirrors:
            # Low resolution and 10 Hz: three extra cameras should not cost much GPU time.
            for key, transform, fov, size in mirror_specs(vehicle):
                camera = self._spawn_camera(vehicle, transform, size, fov, tick=self.mirror_tick)
                camera.listen(lambda image, key=key: self._on_image(image, key))
                self._mirror_cameras[key] = camera
        with self._lock:
            self._frame = None
            self._mirror_frames = {}

    def _destroy_camera(self):
        cameras = list(self._mirror_cameras.values())
        if self._camera is not None:
            cameras.append(self._camera)
        for camera in cameras:
            try:
                camera.stop()
                camera.destroy()
            except RuntimeError:
                pass
        self._camera = None
        self._mirror_cameras = {}

    def _on_image(self, image, key):
        """Camera callback (CARLA worker thread): convert and keep only the latest frame."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        # BGRA -> RGB; copy so the frame does not reference CARLA's buffer after return.
        array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1].copy()
        if key is not None:
            array = array[:, ::-1]   # a mirror shows the scene left-right reversed
        with self._lock:
            if key is None:
                self._frame = array
            else:
                self._mirror_frames[key] = array

    # ----------------------------------------------------------- selection
    def select(self, index):
        if 0 <= index < len(self.fleet.robotaxis) and index != self.selected:
            self.selected = index
            self._pending = None
            self._attach_camera()

    def selected_robotaxi(self):
        if not self.fleet.robotaxis:
            return None
        return self.fleet.robotaxis[self.selected]

    def _next_request(self):
        requests = self.bus.open_requests()
        names = [r.display_name for r in self.fleet.robotaxis]
        if requests:
            current = names[self.selected]
            for request in requests:
                if request.vehicle != current and request.vehicle in names:
                    self.select(names.index(request.vehicle))
                    return
        self.select((self.selected + 1) % max(1, len(names)))

    # -------------------------------------------------------------- actions
    def _press(self, button_id, shift=False):
        """A key press or button click for an action/control."""
        robotaxi = self.selected_robotaxi()
        if robotaxi is None:
            return
        sim_time = self.sim.sim_time
        if button_id == "INJECT":
            self.scenarios.trigger(sim_time, target=robotaxi.display_name,
                                   scenario_id=self._scenario_id())
            return
        if button_id == "SCENARIO":
            ids = self.scenarios.scenario_ids()
            if ids:
                self._scenario_index = (self._scenario_index + 1) % len(ids)
                self.bus.notice("T now injects {}".format(ids[self._scenario_index]))
            return
        if button_id == "CLEAR":
            if shift:
                self.scenarios.clear_all(sim_time)
            elif self._snapshot is not None:
                self.scenarios.clear_for_vehicle(robotaxi, self._snapshot, sim_time)
            return
        if button_id not in robotaxi.available_actions():
            reason = robotaxi.unavailable_reasons().get(button_id, "not available now")
            self.bus.notice("{} {}: {}".format(robotaxi.display_name, button_id, reason))
            self._pending = None
            return
        pending = self._pending
        if (not self.confirm or not NEEDS_CONFIRM.get(button_id, False)
                or (pending and pending[0] == button_id and pending[1] == robotaxi.display_name)):
            self._send(button_id, robotaxi)
        else:
            self._pending = (button_id, robotaxi.display_name, time.perf_counter())

    def _scenario_id(self):
        ids = self.scenarios.scenario_ids()
        if not ids:
            return None
        return ids[self._scenario_index % len(ids)]

    def _send(self, action, robotaxi):
        request = robotaxi.request
        self.bus.send_command(robotaxi.display_name, action, self.sim.sim_time,
                              request.request_id if request else None)
        self.bus.notice("sent {} to {}".format(action, robotaxi.display_name))
        self._pending = None

    def _confirm_pending(self):
        if self._pending is None:
            return
        action, name, _ = self._pending
        robotaxi = self.fleet.by_display_name(name)
        if robotaxi is not None:
            self._send(action, robotaxi)

    # ---------------------------------------------------------------- input
    def handle_events(self, snapshot=None):
        """Process pygame events. Returns False when the user wants to quit."""
        self._snapshot = snapshot
        if self._pending and time.perf_counter() - self._pending[2] > CONFIRM_TIMEOUT_S:
            self._pending = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button_id, rect in self._button_rects.items():
                    if rect.collidepoint(event.pos):
                        self._press(button_id)
                        break
                continue
            if event.type != pygame.KEYDOWN:
                continue
            key = event.key
            shift = bool(event.mod & pygame.KMOD_SHIFT)
            if key == pygame.K_ESCAPE:
                if self._pending:
                    self._pending = None
                    continue
                return False
            if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._confirm_pending()
            elif pygame.K_1 <= key <= pygame.K_9:
                self.select(key - pygame.K_1)
            elif key == pygame.K_TAB:
                self._next_request()
            elif key in FUNCTION_KEYS:
                ids = self.scenarios.scenario_ids()
                index = FUNCTION_KEYS.index(key)
                robotaxi = self.selected_robotaxi()
                if index < len(ids) and robotaxi is not None:
                    self.scenarios.trigger(self.sim.sim_time, target=robotaxi.display_name,
                                           scenario_id=ids[index])
            elif key in KEY_TO_BUTTON:
                self._press(KEY_TO_BUTTON[key], shift)
            elif key == pygame.K_v:
                self.mode = (self.mode + 1) % len(CAMERA_MODES)
                self._attach_camera()
            elif key == pygame.K_m:
                self.mirrors = not self.mirrors
                self._attach_camera()
                self.bus.notice("Mirrors {}".format("on" if self.mirrors else "off"))
            elif key == pygame.K_f:
                self.follow = not self.follow
            elif key == pygame.K_d:
                self.bus.notice("Debug drawing: {}".format(self.drawer.cycle()))
            elif key == pygame.K_h:
                self.show_help = not self.show_help
        return True

    # --------------------------------------------------------------- render
    def render(self):
        now = time.perf_counter()
        elapsed = now - self._last_wall
        self._last_wall = now
        if elapsed > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / elapsed)

        with self._lock:
            frame = self._frame
            mirror_frames = dict(self._mirror_frames)
        if frame is not None:
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            self._display.blit(surface, (0, 0))
        else:
            self._display.fill((20, 20, 24))

        if self.follow:
            self._follow_spectator()
        if self.mirrors:
            self._draw_mirrors(mirror_frames)
        self._draw_fleet_panel()
        self._draw_side_panel()
        self._draw_banner()
        self._draw_action_bar()
        if self.show_help:
            self._draw_help()
        pygame.display.flip()

    def _follow_spectator(self):
        """Put the server window's camera where the viewer's main camera is (front view by
        default), so both windows show the same picture. The server window can only show
        one camera: the mirrors exist in the operator window only."""
        robotaxi = self.selected_robotaxi()
        if robotaxi is None:
            return
        vehicle_transform = robotaxi.vehicle.get_transform()
        offset = camera_transform(CAMERA_MODES[self.mode], robotaxi.vehicle)
        location = vehicle_transform.transform(offset.location)
        rotation = carla.Rotation(pitch=vehicle_transform.rotation.pitch + offset.rotation.pitch,
                                  yaw=vehicle_transform.rotation.yaw + offset.rotation.yaw,
                                  roll=0.0)
        self.sim.world.get_spectator().set_transform(carla.Transform(location, rotation))

    def _panel(self, x, y, w, h, alpha=170):
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, alpha))
        self._display.blit(panel, (x, y))

    def _text(self, text, x, y, color=(235, 235, 235), big=False, max_width=None):
        font = self._font_big if big else self._font
        if max_width is not None and font.size(text)[0] > max_width:
            while text and font.size(text + "..")[0] > max_width:
                text = text[:-1]
            text += ".."
        self._display.blit(font.render(text, True, color), (x, y))

    # -------------------------------------------------------------- mirrors
    def _mirror_rects(self):
        rear_w, rear_h = REAR_MIRROR
        side_w, side_h = SIDE_MIRROR
        side_y = self.height - BAR_HEIGHT - 60 - 84 - side_h
        return {
            "rear": pygame.Rect((self.width - rear_w) // 2, 8, rear_w, rear_h),
            "left": pygame.Rect(10, side_y, side_w, side_h),
            "right": pygame.Rect(self.width - side_w - 10, side_y, side_w, side_h),
        }

    def _draw_mirrors(self, frames):
        for key, rect in self._mirror_rects().items():
            frame = frames.get(key)
            pygame.draw.rect(self._display, (25, 25, 28), rect.inflate(8, 8), border_radius=9)
            if frame is None:
                pygame.draw.rect(self._display, (45, 45, 50), rect, border_radius=6)
                continue
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            self._display.blit(pygame.transform.smoothscale(surface, rect.size), rect)
            pygame.draw.rect(self._display, (90, 90, 96), rect.inflate(4, 4), width=2, border_radius=7)

    def _draw_fleet_panel(self):
        rows = len(self.fleet.robotaxis)
        self._panel(10, 10, PANEL_WIDTH, 50 + rows * 40)
        self._text("FLEET  sim {:7.1f}s  {:4.1f} fps  [{}]".format(
            self.sim.sim_time, self._fps, CAMERA_MODES[self.mode]), 20, 16)
        for index, robotaxi in enumerate(self.fleet.robotaxis):
            marker = ">" if index == self.selected else " "
            color = STATE_COLORS.get(robotaxi.state, (235, 235, 235))
            y = 40 + index * 40
            phase = "drop" if robotaxi.mission.phase == "TO_DROPOFF" else "pick"
            self._text("{}{} {:<6}{:<12}{:3.0f}km/h {} T{} C{}".format(
                marker, index + 1, robotaxi.display_name, robotaxi.state, robotaxi.speed_kmh,
                phase, robotaxi.mission.trips_completed, robotaxi.collisions), 20, y, color,
                max_width=PANEL_WIDTH - 20)
            note = robotaxi.status_note
            via = getattr(robotaxi, "via_site", None)
            if not note and via is not None:
                note = "route via incident site {}@{}".format(via.scenario_id.split("_")[0], via.name)
            if note:
                self._text("   " + note, 20, y + 18, (170, 170, 170), max_width=PANEL_WIDTH - 20)

    def _draw_side_panel(self):
        x = self.width - PANEL_WIDTH - 10
        requests = self.bus.open_requests()
        scenario_lines = self.scenarios.status_lines(self.sim.sim_time)
        notices = list(self.bus.notices)[:6]
        # Stay above the right-hand mirror.
        max_lines = max(4, (self._mirror_rects()["right"].top - 90) // 20) if self.mirrors else 30
        while max(1, len(requests)) + len(scenario_lines) + len(notices) > max_lines and notices:
            notices.pop()
        while max(1, len(requests)) + len(scenario_lines) + len(notices) > max_lines and scenario_lines:
            scenario_lines.pop()
        height = 70 + 20 * (max(1, len(requests)) + len(scenario_lines) + len(notices))
        self._panel(x, 10, PANEL_WIDTH, height)
        y = 16
        self._text("RA REQUESTS", x + 10, y)
        y += 20
        if not requests:
            self._text("  none", x + 10, y, (160, 160, 160))
            y += 20
        for request in requests:
            self._text("  {} {} {} {:4.0f}s".format(
                request.request_id, request.vehicle, request.reason,
                request.waiting_time(self.sim.sim_time)), x + 10, y, (255, 120, 120), max_width=PANEL_WIDTH - 20)
            y += 20
        self._text("SCENARIOS", x + 10, y)
        y += 20
        for line in scenario_lines:
            self._text("  " + line, x + 10, y, (255, 190, 90), max_width=PANEL_WIDTH - 20)
            y += 20
        for notice in notices:
            self._text("  " + notice, x + 10, y, (170, 170, 170), max_width=PANEL_WIDTH - 20)
            y += 20

    def _draw_banner(self):
        robotaxi = self.selected_robotaxi()
        if robotaxi is None:
            return
        y = self.height - BAR_HEIGHT - 60
        if self._pending and self._pending[1] == robotaxi.display_name:
            pygame.draw.rect(self._display, (150, 110, 10), (0, y, self.width, 56))
            self._text("CONFIRM {} for {}?".format(self._pending[0], robotaxi.display_name),
                       16, y + 6, big=True)
            self._text("press ENTER or the same action again; ESC cancels", 16, y + 32)
            return
        if robotaxi.state == "EXECUTING" and robotaxi.maneuver is not None:
            pygame.draw.rect(self._display, (20, 80, 140), (0, y, self.width, 56))
            self._text("{} EXECUTING {}".format(robotaxi.display_name, robotaxi.maneuver.action),
                       16, y + 6, big=True)
            self._text(robotaxi.status_note[:120], 16, y + 32)
            return
        request = robotaxi.request
        if request is None:
            return
        detail = request.detail
        unavailable = robotaxi.unavailable_reasons()
        lines = ["x {}: {}".format(action, unavailable[action])[:118]
                 for action in (ra.ACTION_BYPASS_LEFT, ra.ACTION_BYPASS_RIGHT,
                                ra.ACTION_TURN_LEFT, ra.ACTION_TURN_RIGHT, ra.ACTION_PULL_OVER)
                 if action in unavailable][:4]
        if lines:
            extra = 18 * len(lines) + 6
            self._panel(0, y - extra, self.width, extra, alpha=190)
            for index, line in enumerate(lines):
                self._text(line, 16, y - extra + 4 + index * 18, (200, 170, 170))
        pygame.draw.rect(self._display, (150, 20, 30), (0, y, self.width, 56))
        self._text("{} REQUESTS REMOTE ASSISTANCE: {}   waiting {:.0f}s".format(
            request.vehicle, request.reason, request.waiting_time(self.sim.sim_time)),
            16, y + 6, big=True)
        if "obstacle_type" in detail:
            line = "obstacle {} at {:.1f} m   left lane: {}   right lane: {}".format(
                detail.get("obstacle_type", "?"), detail.get("obstacle_distance_m", -1),
                detail.get("left_lane", "?"), detail.get("right_lane", "?"))
        else:
            line = detail.get("note", "")
        self._text(line, 16, y + 32)

    def _draw_action_bar(self):
        robotaxi = self.selected_robotaxi()
        available = set(robotaxi.available_actions()) if robotaxi else set()
        y = self.height - BAR_HEIGHT
        self._panel(0, y, self.width, BAR_HEIGHT, alpha=200)
        gap = 6
        width = (self.width - gap * (len(BUTTONS) + 1)) // len(BUTTONS)
        self._button_rects = {}
        for index, (button_id, label, _, _) in enumerate(BUTTONS):
            rect = pygame.Rect(gap + index * (width + gap), y + 6, width, BAR_HEIGHT - 12)
            self._button_rects[button_id] = rect
            pending = self._pending and self._pending[0] == button_id
            if pending:
                fill, text = (200, 150, 20), (0, 0, 0)
            elif button_id in available:
                fill, text = (40, 110, 60), (240, 240, 240)
            elif button_id in CONTROL_BUTTONS:
                fill, text = (60, 60, 75), (220, 220, 220)
            else:
                fill, text = (45, 45, 45), (130, 130, 130)
            if button_id == "INJECT":
                label = "T {}".format((self._scenario_id() or "-").split("_")[0])
            pygame.draw.rect(self._display, fill, rect, border_radius=5)
            surface = self._font.render(label, True, text)
            self._display.blit(surface, surface.get_rect(center=rect.center))
        self._draw_tooltip(robotaxi, available)

    def _draw_tooltip(self, robotaxi, available):
        """Reason a grey action button is unavailable, shown while the mouse is over it."""
        if robotaxi is None:
            return
        position = pygame.mouse.get_pos()
        for button_id, rect in self._button_rects.items():
            if button_id in CONTROL_BUTTONS or button_id in available or not rect.collidepoint(position):
                continue
            reason = robotaxi.unavailable_reasons().get(button_id)
            if not reason:
                return
            surface = self._font.render("{}: {}".format(button_id, reason), True, (255, 230, 200))
            box = surface.get_rect()
            box.midbottom = (min(max(rect.centerx, box.width // 2 + 4), self.width - box.width // 2 - 4),
                             rect.top - 8)
            pygame.draw.rect(self._display, (40, 30, 30), box.inflate(12, 8), border_radius=4)
            self._display.blit(surface, box)
            return

    def _draw_help(self):
        height = 20 * len(HELP_LINES) + 12
        y = self.height - BAR_HEIGHT - 60 - height - 6
        self._panel(0, y, self.width, height, alpha=200)
        for index, line in enumerate(HELP_LINES):
            self._text(line, 16, y + 6 + index * 20, (210, 210, 210))

    # -------------------------------------------------------------- cleanup
    def destroy(self):
        self._destroy_camera()
        pygame.quit()
