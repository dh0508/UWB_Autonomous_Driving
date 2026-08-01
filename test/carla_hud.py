"""~/carla/PythonAPI/examples/manual_control.py와 같은 화면(좌측 정보 패널 + 알림 + 도움말)을 재현.

rule_based_test.py의 hero는 키보드가 아니라 규칙 기반 로직이 제어하므로 KeyboardControl은
없고, manual_control.py의 HUD/FadingText/HelpText/CollisionSensor를 옮겨와 표시 항목(서버·클라
FPS, 속도, 나침반, 가속도, 자이로, 좌표, throttle/steer/brake 바, 충돌 히스토리 그래프, 주변 차량)을
원본과 같은 레이아웃으로 그린다. 원본의 F1(HUD 토글)/H·?(도움말 토글)/ESC(종료) 단축키만 그대로 지원한다.
"""

from __future__ import annotations

import collections
import datetime
import math
import weakref

import pygame

HELP_TEXT = """
UME Drive Net — Carla live test

규칙 기반(rule-based) 로직이 hero 차량을 자율주행합니다 (키보드로 직접 조작하지 않음).

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""


def get_actor_display_name(actor, truncate: int = 250) -> str:
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "…") if len(name) > truncate else name


class FadingText:
    def __init__(self, font: pygame.font.Font, dim: tuple[int, int], pos: tuple[int, int]) -> None:
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0.0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text: str, color: tuple[int, int, int] = (255, 255, 255), seconds: float = 2.0) -> None:
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, clock: pygame.time.Clock) -> None:
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(int(500.0 * self.seconds_left))

    def render(self, display: pygame.Surface) -> None:
        display.blit(self.surface, self.pos)


class HelpText:
    def __init__(self, font: pygame.font.Font, width: int, height: int, text: str = HELP_TEXT) -> None:
        lines = text.split("\n")
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
        self.surface.set_alpha(220)
        self._render = False

    def toggle(self) -> None:
        self._render = not self._render

    def render(self, display: pygame.Surface) -> None:
        if self._render:
            display.blit(self.surface, self.pos)


class CollisionSensor:
    """manual_control.CollisionSensor와 동일 — HUD 충돌 히스토리 그래프용."""

    def __init__(self, parent_actor, hud: "HUD", carla_module) -> None:
        self.sensor = None
        self.history: list[tuple[int, float]] = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(bp, carla_module.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self) -> dict[int, float]:
        history: dict[int, float] = collections.defaultdict(float)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event) -> None:
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification(f"Collision with {actor_type!r}")
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

    def destroy(self) -> None:
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


class LaneInvasionSensor:
    """manual_control.LaneInvasionSensor와 동일 — 차선(중앙선 등) 침범 시 HUD 알림."""

    def __init__(self, parent_actor, hud: "HUD", carla_module) -> None:
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        self.sensor = world.spawn_actor(bp, carla_module.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event) -> None:
        self = weak_self()
        if not self:
            return
        lane_types = {x.type for x in event.crossed_lane_markings}
        text = [str(x).split()[-1] for x in lane_types]
        self.hud.notification(f"Crossed line {' and '.join(text)}")

    def destroy(self) -> None:
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


class HUD:
    """manual_control.HUD와 같은 레이아웃의 좌측 반투명 정보 패널 + 알림 + 도움말."""

    def __init__(self, width: int, height: int) -> None:
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if "mono" in x]
        default_font = "ubuntumono"
        mono = default_font if default_font in fonts else (fonts[0] if fonts else None)
        mono = pygame.font.match_font(mono) if mono else pygame.font.get_default_font()
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0.0
        self.frame = 0
        self.simulation_time = 0.0
        self._show_info = True
        self._info_text: list = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, world_snapshot) -> None:
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = world_snapshot.frame
        self.simulation_time = world_snapshot.elapsed_seconds

    def toggle_info(self) -> None:
        self._show_info = not self._show_info

    def notification(self, text: str, seconds: float = 2.0) -> None:
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text: str) -> None:
        self._notifications.set_text(f"Error: {text}", (255, 0, 0))

    def tick(
        self,
        clock: pygame.time.Clock,
        player,
        map_name: str,
        location: tuple[float, float, float],
        speed: float,
        yaw_deg: float,
        accel: tuple[float, float, float],
        yaw_rate_deg: float,
        control: dict[str, float] | None,
        collision_sensor: CollisionSensor,
        route_remaining_m: float | None,
        status: str,
    ) -> None:
        self._notifications.tick(clock)
        if not self._show_info:
            return

        compass = yaw_deg % 360.0
        heading = "N" if compass > 270.5 or compass < 89.5 else ""
        heading += "S" if 90.5 < compass < 269.5 else ""
        heading += "E" if 0.5 < compass < 179.5 else ""
        heading += "W" if 180.5 < compass < 359.5 else ""

        colhist = collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]

        vehicles = player.get_world().get_actors().filter("vehicle.*")

        self._info_text = [
            "Server:  % 16.0f FPS" % self.server_fps,
            "Client:  % 16.0f FPS" % clock.get_fps(),
            "",
            "Vehicle: % 20s" % get_actor_display_name(player, truncate=20),
            "Map:     % 20s" % map_name,
            "Simulation time: % 12s" % datetime.timedelta(seconds=int(self.simulation_time)),
            "",
            "Status:  % 20s" % status,
            "Speed:   % 15.0f km/h" % (3.6 * speed),
            "Compass:% 17.0f\N{DEGREE SIGN} % 2s" % (compass, heading),
            "Accelero: (%5.1f,%5.1f,%5.1f)" % accel,
            "Gyroscop: (%5.1f,%5.1f,%5.1f)" % (0.0, 0.0, yaw_rate_deg),
            "Location:% 20s" % ("(% 5.1f, % 5.1f)" % (location[0], location[1])),
            "Height:  % 18.0f m" % location[2],
            "",
        ]
        if control is not None:
            self._info_text += [
                ("Throttle:", control["throttle"], 0.0, 1.0),
                ("Steer:", control["steer"], -1.0, 1.0),
                ("Brake:", control["brake"], 0.0, 1.0),
            ]
        if route_remaining_m is not None:
            self._info_text += ["Route left: % 11.0f m" % route_remaining_m]
        self._info_text += [
            "",
            "Collision:",
            collision,
            "",
            "Number of vehicles: % 8d" % len(vehicles),
        ]
        if len(vehicles) > 1:
            self._info_text += ["Nearby vehicles:"]

            def distance(loc) -> float:
                return math.sqrt(
                    (loc.x - location[0]) ** 2 + (loc.y - location[1]) ** 2 + (loc.z - location[2]) ** 2
                )

            others = [(distance(v.get_location()), v) for v in vehicles if v.id != player.id]
            for d, vehicle in sorted(others, key=lambda item: item[0]):
                if d > 200.0:
                    break
                self._info_text.append("% 4dm %s" % (d, get_actor_display_name(vehicle, truncate=22)))

    def render(self, display: pygame.Surface) -> None:
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                    f = (item[1] - item[2]) / (item[3] - item[2])
                    if item[2] < 0.0:
                        rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                    else:
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)
