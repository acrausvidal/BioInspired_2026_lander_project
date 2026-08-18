__credits__ = ["Andrea PIERRÉ", "TU Delft AE4350"]

import math
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed. Install via `pip install "gymnasium[box2d]"`'
    ) from e

if TYPE_CHECKING:
    import pygame


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Initial random disturbance impulse

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = 4

VIEWPORT_W = 600
VIEWPORT_H = 400

# Physical Mass and Fuel Constants
INITIAL_FUEL_CAPACITY = 100.0  # Fuel units
INITIAL_DENSITY = 5.0          # Initial Box2D body density (wet mass ~4.82 kg)
DRY_DENSITY = 2.5              # Dry Box2D body density at zero fuel (dry mass ~2.41 kg)
MAIN_FUEL_CONSUMPTION = 0.40   # Fuel consumed per frame at full main throttle (1.0)
SIDE_FUEL_CONSUMPTION = 0.05   # Fuel consumed per frame at full side throttle (1.0)
FUEL_PENALTY_COEFF = 0.50      # Additional reward penalty multiplier per unit of fuel consumed


class ContactDetector(contactListener):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.lander == contact.fixtureA.body
            or self.env.lander == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class CustomLunarLanderContinuous(gym.Env, EzPickle):
    r"""
    ## Bio-Inspired Intelligence Continuous Lunar Lander with Mass-Varying Physics & Fuel Constraints
    
    ### Dynamic Mass Model:
    The vehicle starts with initial fuel capacity `initial_fuel = 100.0` and density $\rho_0 = 5.0$.
    As propellant is consumed by the continuous thrusters, the vehicle mass dynamically decreases:
    $$\rho(t) = \rho_{\text{dry}} + (\rho_0 - \rho_{\text{dry}}) \cdot \frac{F(t)}{F_0}$$
    The Box2D physics engine mass tensor is updated at every physics step via `body.ResetMassData()`.
    
    ### Observation Space (9 Dimensions):
    0: Horizontal coordinate $x \in [-2.5, 2.5]$
    1: Vertical coordinate $y \in [-2.5, 2.5]$
    2: Horizontal velocity $v_x \in [-10.0, 10.0]$
    3: Vertical velocity $v_y \in [-10.0, 10.0]$
    4: Orientation angle $\theta \in [-2\pi, 2\pi]$
    5: Angular velocity $\omega \in [-10.0, 10.0]$
    6: Left leg ground contact boolean $\{0.0, 1.0\}$
    7: Right leg ground contact boolean $\{0.0, 1.0\}$
    8: Normalized remaining fuel fraction $f(t) = F(t) / F_0 \in [0.0, 1.0]$
    
    ### Action Space (Continuous 2D Box [-1, 1]):
    - Action[0] (Main Engine): $u_{\text{main}} \le 0 \implies \text{off}$; $u_{\text{main}} \in (0, 1] \implies$ throttle $[0.5, 1.0]$
    - Action[1] (Side Engines): $u_{\text{side}} \in [-0.5, 0.5] \implies \text{off}$; $|u_{\text{side}}| > 0.5 \implies$ directional throttle $[0.5, 1.0]$
    
    ### Terminal Conditions:
    1. Safe Landing: Vehicle comes to rest on landing pad with legs touching (`reward = +100`).
    2. Crash: Body collision with ground or vehicle out of bounds (`reward = -100`).
    3. Fuel Exhaustion: Vehicle depletes fuel before reaching a safe landing state (`reward = -100`).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        initial_fuel: float = INITIAL_FUEL_CAPACITY,
        initial_density: float = INITIAL_DENSITY,
        dry_density: float = DRY_DENSITY,
        fuel_penalty_coeff: float = FUEL_PENALTY_COEFF,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            gravity,
            enable_wind,
            wind_power,
            turbulence_power,
            initial_fuel,
            initial_density,
            dry_density,
            fuel_penalty_coeff,
        )

        assert -12.0 < gravity < 0.0, f"gravity ({gravity}) must be between -12 and 0"
        self.gravity = gravity
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power
        self.enable_wind = enable_wind

        # Fuel and Mass Configuration
        self.initial_fuel = float(initial_fuel)
        self.fuel = float(initial_fuel)
        self.initial_density = float(initial_density)
        self.dry_density = float(dry_density)
        self.fuel_penalty_coeff = float(fuel_penalty_coeff)

        self.screen: Optional["pygame.Surface"] = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon = None
        self.lander: Optional[Box2D.b2Body] = None
        self.legs = []
        self.particles = []
        self.prev_shaping = None
        self.total_fuel_consumed = 0.0

        # Continuous Action Space: [Main Thruster, Lateral Thrusters]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 9-dimensional observation space (includes normalized fuel)
        low = np.array(
            [
                -2.5,          # x
                -2.5,          # y
                -10.0,         # vx
                -10.0,         # vy
                -2 * math.pi,  # angle
                -10.0,         # angular velocity
                0.0,           # leg 1 contact
                0.0,           # leg 2 contact
                0.0,           # normalized fuel
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                2.5,           # x
                2.5,           # y
                10.0,          # vx
                10.0,          # vy
                2 * math.pi,   # angle
                10.0,          # angular velocity
                1.0,           # leg 1 contact
                1.0,           # leg 2 contact
                1.0,           # normalized fuel
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.render_mode = render_mode

    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        if self.lander is not None:
            self.world.DestroyBody(self.lander)
            self.lander = None
        if len(self.legs) == 2:
            self.world.DestroyBody(self.legs[0])
            self.world.DestroyBody(self.legs[1])
            self.legs = []

    def _update_mass(self):
        """Dynamically update Box2D lander mass and inertia tensor based on remaining fuel."""
        if self.lander is None or not self.lander.fixtures:
            return
        fuel_fraction = np.clip(self.fuel / self.initial_fuel, 0.0, 1.0)
        current_density = self.dry_density + (self.initial_density - self.dry_density) * fuel_fraction
        
        for fixture in self.lander.fixtures:
            fixture.density = current_density
        self.lander.ResetMassData()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._destroy()

        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        self.fuel = self.initial_fuel
        self.total_fuel_consumed = 0.0

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create Lunar Surface Terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        # Create Lander body with initial wet density
        initial_y = VIEWPORT_H / SCALE
        initial_x = VIEWPORT_W / SCALE / 2
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=self.initial_density,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0,
            ),
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)

        # Initial disturbance impulse
        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        if self.enable_wind:
            self.wind_idx = self.np_random.integers(-9999, 9999)
            self.torque_idx = self.np_random.integers(-9999, 9999)

        # Create Lander Landing Gear (Legs)
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs
        self._update_mass()

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        return obs, {"fuel_remaining": self.fuel, "lander_mass": self.lander.mass}

    def _get_obs(self) -> np.ndarray:
        pos = self.lander.position
        vel = self.lander.linearVelocity
        fuel_normalized = float(np.clip(self.fuel / self.initial_fuel, 0.0, 1.0))

        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            fuel_normalized,
        ]
        return np.array(state, dtype=np.float32)

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particle):
        while self.particles and (all_particle or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.lander is not None, "Call reset() before calling step()"

        # Wind & Turbulence Simulation
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter((wind_mag, 0.0), True)

            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + (math.sin(math.pi * 0.01 * self.torque_idx))
                )
                * self.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(torque_mag, True)

        # Continuous Action clipping
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Check Fuel Availability
        has_fuel = self.fuel > 0.0
        step_fuel_consumed = 0.0

        # Orientation Vectors
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if has_fuel and action[0] > 0.0:
            m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
            fuel_burn = m_power * MAIN_FUEL_CONSUMPTION
            fuel_burn = min(self.fuel, fuel_burn)
            self.fuel -= fuel_burn
            step_fuel_consumed += fuel_burn
            
            # Scale effective thrust if fuel depleted mid-step
            eff_power = m_power * (fuel_burn / (m_power * MAIN_FUEL_CONSUMPTION))

            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)

            if self.render_mode is not None:
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], eff_power)
                p.ApplyLinearImpulse(
                    (ox * MAIN_ENGINE_POWER * eff_power, oy * MAIN_ENGINE_POWER * eff_power),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * eff_power, -oy * MAIN_ENGINE_POWER * eff_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if self.fuel > 0.0 and np.abs(action[1]) > 0.5:
            direction = np.sign(action[1])
            s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
            fuel_burn = s_power * SIDE_FUEL_CONSUMPTION
            fuel_burn = min(self.fuel, fuel_burn)
            self.fuel -= fuel_burn
            step_fuel_consumed += fuel_burn

            eff_power = s_power * (fuel_burn / (s_power * SIDE_FUEL_CONSUMPTION))

            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )

            if self.render_mode is not None:
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], eff_power)
                p.ApplyLinearImpulse(
                    (ox * SIDE_ENGINE_POWER * eff_power, oy * SIDE_ENGINE_POWER * eff_power),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * eff_power, -oy * SIDE_ENGINE_POWER * eff_power),
                impulse_pos,
                True,
            )

        self.total_fuel_consumed += step_fuel_consumed

        # Dynamically Update Box2D Vehicle Mass & Inertia
        self._update_mass()

        # Step Box2D Physics Engine
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        state = self._get_obs()

        # Reward Shaping
        reward = 0.0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Fuel Consumption Penalization
        reward -= (m_power * 0.30 + s_power * 0.03)
        reward -= step_fuel_consumed * self.fuel_penalty_coeff

        # Termination Evaluation
        terminated = False
        out_of_fuel = (self.fuel <= 0.0)
        is_safe_landing = (not self.lander.awake) and (self.legs[0].ground_contact and self.legs[1].ground_contact)

        if self.game_over or abs(state[0]) >= 1.0:
            # Vehicle crashed into terrain or went out of bounds
            terminated = True
            reward = -100.0
        elif not self.lander.awake:
            # Vehicle came to a complete rest
            terminated = True
            reward = +100.0
        elif out_of_fuel:
            # Vehicle depleted all fuel while still in flight
            terminated = True
            reward = -100.0

        info = {
            "fuel_remaining": self.fuel,
            "fuel_consumed": self.total_fuel_consumed,
            "lander_mass": float(self.lander.mass),
            "lander_inertia": float(self.lander.inertia),
            "is_safe_landing": is_safe_landing,
            "out_of_fuel": out_of_fuel,
        }

        if self.render_mode == "human":
            self.render()

        return state, reward, terminated, False, info

    def render(self):
        if self.render_mode is None:
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed. Run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )

        self._clean_particles(False)

        for p in self.sky_polys:
            scaled_poly = [(coord[0] * SCALE, coord[1] * SCALE) for coord in p]
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(
                        self.surf, color=obj.color2, points=path, closed=True
                    )

                for x in [self.helipad_x1, self.helipad_x2]:
                    x = x * SCALE
                    flagy1 = self.helipad_y * SCALE
                    flagy2 = flagy1 + 50
                    pygame.draw.line(
                        self.surf,
                        color=(255, 255, 255),
                        start_pos=(x, flagy1),
                        end_pos=(x, flagy2),
                        width=1,
                    )
                    pygame.draw.polygon(
                        self.surf,
                        color=(204, 204, 0),
                        points=[
                            (x, flagy2),
                            (x, flagy2 - 10),
                            (x + 25, flagy2 - 5),
                        ],
                    )
                    gfxdraw.aapolygon(
                        self.surf,
                        [
                            (x, flagy2),
                            (x, flagy2 - 10),
                            (x + 25, flagy2 - 5),
                        ],
                        (204, 204, 0),
                    )

        # Flip surface vertically so Box2D (y=0 at bottom) displays upright in Pygame
        self.surf = pygame.transform.flip(self.surf, False, True)

        # Draw HUD for remaining fuel and vehicle mass (after flip so text is upright)
        if pygame.font.get_init():
            font = pygame.font.SysFont("Arial", 14)
            fuel_pct = (self.fuel / self.initial_fuel) * 100
            current_mass = self.lander.mass if self.lander else 0.0
            fuel_text = font.render(f"Propellant: {fuel_pct:4.1f}% | Vehicle Mass: {current_mass:4.2f} kg", True, (20, 20, 20))
            self.surf.blit(fuel_text, (10, 10))

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


# Gymnasium registration
gym.register(
    id="CustomLunarLanderContinuous-v0",
    entry_point="src.custom_lander:CustomLunarLanderContinuous",
    max_episode_steps=1000,
    reward_threshold=200,
)
