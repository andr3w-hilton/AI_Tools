"""
Orb Arena - Multiplayer WebSocket Game Server
A competitive arena game where players control orbs, collect energy, and consume smaller players.
"""

import asyncio
import json
import random
import math
import time

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print("Using uvloop (faster async)")
except ImportError:
    pass  # Falls back to default asyncio loop

import websockets
from websockets.exceptions import ConnectionClosed
from dataclasses import dataclass
from typing import Dict, Set, Optional
import colorsys
import socket
import http.server
import threading
import os
import re

# Game configuration
WORLD_WIDTH = 2000
WORLD_HEIGHT = 2000
INITIAL_RADIUS = 20
MAX_RADIUS = 150
MIN_RADIUS = 10
BASE_SPEED = 14
SPEED_SCALING = 0.3  # Higher = more speed difference between small and large
ENERGY_ORB_COUNT = 100
ENERGY_ORB_VALUE = 2
ENERGY_ORB_RADIUS = 8
SPIKE_ORB_COUNT = 15  # Evil spike orbs
SPIKE_ORB_RADIUS = 12
GOLDEN_ORB_COUNT = 5  # Rare golden orbs
GOLDEN_ORB_VALUE = 10  # 5x energy orb value
GOLDEN_ORB_RADIUS = 12
TICK_RATE = 1 / 30  # 30 FPS (reduced from 60 for memory efficiency)
SHRINK_RATE = 0.02  # Slowly shrink over time
CONSUME_RATIO = 1.2  # Must be 20% larger to consume another player

# Boost/Dash configuration
BOOST_SPEED_MULTIPLIER = 2.5
BOOST_DURATION = 0.25  # seconds
BOOST_COOLDOWN = 3.0  # seconds
BOOST_MASS_COST = 3  # radius cost to boost

# Projectile/Shooting configuration
PROJECTILE_SPEED = 25
PROJECTILE_RADIUS = 5
PROJECTILE_LIFETIME = 2.0  # seconds
PROJECTILE_DAMAGE = 10  # radius removed from target
PROJECTILE_COST = 5  # radius cost to shooter
PROJECTILE_COOLDOWN = 0.5  # seconds between shots
PROJECTILE_MIN_RADIUS = 25  # must be this big to shoot

# Critical mass configuration
CRITICAL_MASS_THRESHOLD = 100  # radius threshold to start timer
CRITICAL_MASS_TIMER = 30.0  # seconds before explosion

# Power-up configuration
POWERUP_COUNT = 2  # max on map at once
POWERUP_RADIUS = 14
POWERUP_RESPAWN_DELAY = 30.0  # seconds before replacement spawns
POWERUP_TYPES = ["shield", "rapid_fire", "magnet", "phantom"]
POWERUP_DURATIONS = {"shield": 5.0, "rapid_fire": 5.0, "magnet": 8.0, "phantom": 5.0}
MAGNET_RANGE = 300  # radius for magnet pull
MAGNET_STRENGTH = 10  # speed orbs move toward player

# Respawn invincibility
RESPAWN_INVINCIBILITY = 3.0  # seconds

# Kill feed
KILL_FEED_MAX = 5  # max messages to show
KILL_FEED_DURATION = 5.0  # seconds before message expires

# Walls/Obstacles
WALL_COUNT = 8

# ── Natural Disaster Configuration ──
DISASTER_MIN_INTERVAL = 600.0  # 10 minutes minimum between disasters
DISASTER_MAX_INTERVAL = 900.0  # 15 minutes maximum between disasters
DISASTER_WARNING_TIME = 5.0    # seconds of warning before disaster hits
DISASTER_MIN_PLAYERS = 2       # need at least 2 players to trigger
DISASTER_SETTLE_TIME = 120.0   # 2 min grace period after lobby first fills

# Black Hole
BLACK_HOLE_DURATION = 15.0
BLACK_HOLE_MAX_RADIUS = 80     # visual/kill radius at full size
BLACK_HOLE_PULL_RANGE = 500    # gravitational pull range
BLACK_HOLE_PULL_STRENGTH = 18  # base pull speed (scaled by distance)
BLACK_HOLE_MASS_FACTOR = 0.7   # smaller players pulled harder (inverse mass)

# Meteor Shower
METEOR_SHOWER_DURATION = 10.0
METEOR_INTERVAL = 0.15         # seconds between meteor strikes
METEOR_DAMAGE = 8              # radius removed on hit
METEOR_BLAST_RADIUS = 40       # area of effect per meteor
METEOR_COUNT_PER_WAVE = 3      # meteors per interval tick

# Fog of War
FOG_DURATION = 15.0
FOG_VISIBILITY_RADIUS = 300    # pixels around player

# Feeding Frenzy
FRENZY_DURATION = 10.0
FRENZY_ORB_COUNT = 250         # orbs spawned at start

# Supernova
SUPERNOVA_RADIUS = 600         # blast radius from center
SUPERNOVA_MASS_LOSS_MIN = 0.20 # 20% mass loss
SUPERNOVA_MASS_LOSS_MAX = 0.30 # 30% mass loss

# Earthquake
EARTHQUAKE_DURATION = 3.0      # wall transition time


@dataclass
class EnergyOrb:
    id: str
    x: float
    y: float
    radius: float = ENERGY_ORB_RADIUS
    color: str = "#00ff88"

    def to_dict(self):
        return {"id": self.id, "x": round(self.x, 1), "y": round(self.y, 1), "radius": self.radius, "color": self.color}


@dataclass
class SpikeOrb:
    id: str
    x: float
    y: float
    radius: float = SPIKE_ORB_RADIUS
    color: str = "#ff2266"

    def to_dict(self):
        return {"id": self.id, "x": round(self.x, 1), "y": round(self.y, 1), "radius": self.radius, "color": self.color}


@dataclass
class GoldenOrb:
    id: str
    x: float
    y: float
    radius: float = GOLDEN_ORB_RADIUS
    color: str = "#ffd700"

    def to_dict(self):
        return {"id": self.id, "x": round(self.x, 1), "y": round(self.y, 1), "radius": self.radius, "color": self.color}


@dataclass
class PowerUpOrb:
    id: str
    x: float
    y: float
    radius: float = POWERUP_RADIUS
    color: str = "#dd44ff"

    def to_dict(self):
        return {"id": self.id, "x": round(self.x, 1), "y": round(self.y, 1), "radius": self.radius, "color": self.color}


@dataclass
class Projectile:
    id: str
    owner_id: str
    x: float
    y: float
    dx: float  # normalized direction
    dy: float
    radius: float = PROJECTILE_RADIUS
    color: str = "#ffffff"
    created_at: float = 0.0

    def to_dict(self):
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "radius": self.radius,
            "color": self.color
        }


@dataclass
class Wall:
    id: str
    x: float
    y: float
    width: float
    height: float
    color: str = "#334455"

    def to_dict(self):
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "color": self.color
        }


@dataclass
class Meteor:
    x: float
    y: float
    radius: float = METEOR_BLAST_RADIUS
    impact_time: float = 0.0  # when it lands

    def to_dict(self):
        return {"x": round(self.x, 1), "y": round(self.y, 1), "radius": self.radius}


@dataclass
class BlackHole:
    x: float
    y: float
    current_radius: float = 5.0
    max_radius: float = BLACK_HOLE_MAX_RADIUS

    def to_dict(self):
        return {"x": round(self.x, 1), "y": round(self.y, 1), "radius": round(self.current_radius, 1)}


@dataclass
class Player:
    id: str
    name: str
    x: float
    y: float
    radius: float
    color: str
    target_x: float
    target_y: float
    score: int = 0
    alive: bool = True
    # Boost tracking
    boost_cooldown_until: float = 0
    boost_active_until: float = 0
    # Invincibility tracking
    invincible_until: float = 0
    # Shooting tracking
    shoot_cooldown_until: float = 0
    # Critical mass tracking
    critical_mass_start: float = 0  # timestamp when threshold crossed, 0 = inactive
    # Power-up tracking
    active_powerup: str = ""
    powerup_until: float = 0

    def to_dict(self, current_time: float):
        critical_mass_active = self.critical_mass_start > 0
        critical_mass_remaining = 0
        if critical_mass_active:
            elapsed = current_time - self.critical_mass_start
            critical_mass_remaining = max(0, CRITICAL_MASS_TIMER - elapsed)
        return {
            "id": self.id,
            "name": self.name,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "radius": round(self.radius, 1),
            "color": self.color,
            "score": self.score,
            "alive": self.alive,
            "is_boosting": current_time < self.boost_active_until,
            "boost_ready": current_time >= self.boost_cooldown_until,
            "is_invincible": current_time < self.invincible_until,
            "shoot_ready": current_time >= self.shoot_cooldown_until,
            "critical_mass_active": critical_mass_active,
            "critical_mass_remaining": round(critical_mass_remaining, 1),
            "active_powerup": self.active_powerup if current_time < self.powerup_until else "",
            "powerup_remaining": round(max(0, self.powerup_until - current_time), 1) if self.active_powerup else 0
        }

    def get_speed(self, current_time: float):
        # Larger players move slower, smaller players are much faster
        base = BASE_SPEED * (INITIAL_RADIUS / self.radius) ** SPEED_SCALING
        # Apply boost multiplier if active
        if current_time < self.boost_active_until:
            return base * BOOST_SPEED_MULTIPLIER
        return base

    def check_invincible(self, current_time: float):
        return current_time < self.invincible_until


DISASTER_TYPES = ["black_hole", "meteor_shower", "fog_of_war", "feeding_frenzy", "supernova", "earthquake"]


def safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to a finite float, clamped to world bounds."""
    try:
        f = float(value)
        if not math.isfinite(f):
            return default
        # Clamp to reasonable world bounds to prevent absurd values
        return max(-1000, min(max(WORLD_WIDTH, WORLD_HEIGHT) + 1000, f))
    except (TypeError, ValueError):
        return default


def sanitize_name(raw: str) -> str:
    """Sanitize a player name: strip HTML/control chars, collapse whitespace, limit length."""
    # Strip HTML tags
    name = re.sub(r'<[^>]*>', '', raw)
    # Strip control characters and zero-width chars
    name = re.sub(r'[\x00-\x1f\x7f-\x9f\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]', '', name)
    # Collapse whitespace
    name = ' '.join(name.split())
    # Limit length
    name = name[:15].strip()
    return name if name else "Anonymous"


class DisasterManager:
    """Manages natural disaster scheduling and execution."""

    def __init__(self, game):
        self.game = game
        self.active_disaster: Optional[str] = None
        self.disaster_start: float = 0
        self.disaster_end: float = 0
        self.warning_active: bool = False
        self.warning_type: str = ""
        self.warning_start: float = 0
        # Scheduling — timer is paused until enough players join
        self.next_disaster_time: float = 0  # 0 = not scheduled yet
        self.lobby_ready_since: float = 0   # when player count first hit minimum
        self.timer_paused: bool = True       # paused until lobby is ready
        # Black hole state
        self.black_hole: Optional[BlackHole] = None
        # Meteor shower state
        self.meteors: list = []  # active meteor impact markers
        self.last_meteor_time: float = 0
        # Fog of war state
        self.fog_active: bool = False
        # Feeding frenzy state
        self.frenzy_orb_ids: list = []  # track frenzy orbs for cleanup
        # Supernova state
        self.supernova_x: float = 0
        self.supernova_y: float = 0
        self.supernova_triggered: bool = False
        self.supernova_time: float = 0  # when the flash happened
        # Earthquake state
        self.earthquake_progress: float = 0  # 0..1
        self.earthquake_old_walls: list = []
        self.earthquake_new_walls: list = []

    def _player_count(self) -> int:
        return len(self.game.players)

    def tick(self, current_time: float):
        """Called every game tick."""
        player_count = self._player_count()

        # ── Lobby readiness / timer management ──
        if player_count < DISASTER_MIN_PLAYERS:
            # Not enough players — pause timer and reset lobby readiness
            if not self.timer_paused:
                self.timer_paused = True
                self.next_disaster_time = 0
                self.lobby_ready_since = 0
            # If a warning was queued but players left, cancel it
            if self.warning_active and not self.active_disaster:
                self.warning_active = False
                self.warning_type = ""
            return

        # Enough players are present
        if self.timer_paused:
            # Lobby just became ready — start settle period
            self.lobby_ready_since = current_time
            self.timer_paused = False
            # Schedule first disaster after settle time + random interval
            self.next_disaster_time = (
                current_time + DISASTER_SETTLE_TIME
                + random.uniform(DISASTER_MIN_INTERVAL, DISASTER_MAX_INTERVAL)
            )
            return

        # Still in settle period — don't trigger yet
        if current_time - self.lobby_ready_since < DISASTER_SETTLE_TIME:
            return

        # ── Warning phase ──
        if not self.active_disaster and not self.warning_active:
            if self.next_disaster_time > 0 and current_time >= self.next_disaster_time:
                self.warning_type = random.choice(DISASTER_TYPES)
                self.warning_active = True
                self.warning_start = current_time
            return

        if self.warning_active and not self.active_disaster:
            if current_time - self.warning_start >= DISASTER_WARNING_TIME:
                self._start_disaster(self.warning_type, current_time)
                self.warning_active = False
            return

        # ── Active disaster tick ──
        if self.active_disaster:
            if current_time >= self.disaster_end:
                self._end_disaster(current_time)
            else:
                self._tick_disaster(current_time)

    def _start_disaster(self, dtype: str, current_time: float):
        self.active_disaster = dtype

        if dtype == "black_hole":
            self.disaster_start = current_time
            self.disaster_end = current_time + BLACK_HOLE_DURATION
            x, y = self.game.find_safe_orb_position(BLACK_HOLE_MAX_RADIUS)
            self.black_hole = BlackHole(x=x, y=y)

        elif dtype == "meteor_shower":
            self.disaster_start = current_time
            self.disaster_end = current_time + METEOR_SHOWER_DURATION
            self.meteors = []
            self.last_meteor_time = current_time

        elif dtype == "fog_of_war":
            self.disaster_start = current_time
            self.disaster_end = current_time + FOG_DURATION
            self.fog_active = True

        elif dtype == "feeding_frenzy":
            self.disaster_start = current_time
            self.disaster_end = current_time + FRENZY_DURATION
            self._spawn_frenzy_orbs()

        elif dtype == "supernova":
            # Supernova is instant — no duration, just a flash
            self.supernova_x = random.uniform(200, WORLD_WIDTH - 200)
            self.supernova_y = random.uniform(200, WORLD_HEIGHT - 200)
            self.supernova_triggered = True
            self.supernova_time = current_time
            self._apply_supernova()
            # Give 3 seconds for the visual to play out
            self.disaster_start = current_time
            self.disaster_end = current_time + 3.0

        elif dtype == "earthquake":
            self.disaster_start = current_time
            self.disaster_end = current_time + EARTHQUAKE_DURATION
            self.earthquake_progress = 0
            # Save current wall positions
            self.earthquake_old_walls = [
                {"id": w.id, "x": w.x, "y": w.y, "width": w.width, "height": w.height}
                for w in self.game.walls.values()
            ]
            # Generate new random wall positions
            self.earthquake_new_walls = self._generate_new_wall_positions()

    def _tick_disaster(self, current_time: float):
        if self.active_disaster == "black_hole":
            self._tick_black_hole(current_time)
        elif self.active_disaster == "meteor_shower":
            self._tick_meteor_shower(current_time)
        elif self.active_disaster == "earthquake":
            self._tick_earthquake(current_time)
        # fog_of_war, feeding_frenzy, supernova have no per-tick server logic beyond their setup

    def _end_disaster(self, current_time: float):
        dtype = self.active_disaster

        if dtype == "black_hole":
            # Collapse — scatter energy orbs outward from center
            if self.black_hole:
                for _ in range(30):
                    angle = random.uniform(0, math.pi * 2)
                    dist = random.uniform(50, 300)
                    ox = self.black_hole.x + math.cos(angle) * dist
                    oy = self.black_hole.y + math.sin(angle) * dist
                    ox = max(50, min(WORLD_WIDTH - 50, ox))
                    oy = max(50, min(WORLD_HEIGHT - 50, oy))
                    self.game.orb_counter += 1
                    orb_id = f"orb_{self.game.orb_counter}"
                    self.game.energy_orbs[orb_id] = EnergyOrb(id=orb_id, x=ox, y=oy)
                self.game._energy_orbs_cache = None
            self.black_hole = None

        elif dtype == "meteor_shower":
            self.meteors = []

        elif dtype == "fog_of_war":
            self.fog_active = False

        elif dtype == "feeding_frenzy":
            # Remove remaining frenzy orbs
            for orb_id in self.frenzy_orb_ids:
                if orb_id in self.game.energy_orbs:
                    del self.game.energy_orbs[orb_id]
            self.frenzy_orb_ids = []
            self.game._energy_orbs_cache = None

        elif dtype == "supernova":
            self.supernova_triggered = False

        elif dtype == "earthquake":
            # Snap walls to final positions
            self._finalize_earthquake()
            self.earthquake_progress = 0

        self.active_disaster = None
        self.next_disaster_time = current_time + random.uniform(DISASTER_MIN_INTERVAL, DISASTER_MAX_INTERVAL)

    # ── Black Hole ──

    def _tick_black_hole(self, current_time: float):
        bh = self.black_hole
        if not bh:
            return
        elapsed = current_time - self.disaster_start
        progress = min(1.0, elapsed / BLACK_HOLE_DURATION)
        # Grow over time
        bh.current_radius = 5 + (bh.max_radius - 5) * progress

        # Pull players
        for player in self.game.players.values():
            if not player.alive:
                continue
            # Shield and invincibility don't save you from gravity (but do from kill)
            dx = bh.x - player.x
            dy = bh.y - player.y
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1:
                dist_sq = 1
            dist = math.sqrt(dist_sq)

            if dist < BLACK_HOLE_PULL_RANGE:
                # Pull strength: increases as you get closer, lighter players pulled harder
                mass_factor = (INITIAL_RADIUS / player.radius) ** BLACK_HOLE_MASS_FACTOR
                proximity_factor = 1.0 - (dist / BLACK_HOLE_PULL_RANGE)
                pull = BLACK_HOLE_PULL_STRENGTH * proximity_factor * mass_factor * progress
                player.x += (dx / dist) * pull
                player.y += (dy / dist) * pull

                # Kill if dragged into center
                if dist < bh.current_radius * 0.5:
                    if not player.check_invincible(current_time):
                        has_shield = player.active_powerup == "shield" and current_time < player.powerup_until
                        if not has_shield:
                            player.alive = False
                            player.score = 0
                            self.game.add_kill("Black Hole", player.name)

        # Pull energy orbs toward center
        orb_moved = False
        for orb in self.game.energy_orbs.values():
            dx = bh.x - orb.x
            dy = bh.y - orb.y
            dist = math.sqrt(dx * dx + dy * dy)
            if 0 < dist < BLACK_HOLE_PULL_RANGE:
                pull = 8 * (1.0 - dist / BLACK_HOLE_PULL_RANGE) * progress
                orb.x += (dx / dist) * pull
                orb.y += (dy / dist) * pull
                orb_moved = True

        # Consume orbs that reach center
        orbs_consumed = [oid for oid, orb in self.game.energy_orbs.items()
                         if math.sqrt((orb.x - bh.x)**2 + (orb.y - bh.y)**2) < bh.current_radius * 0.5]
        for oid in orbs_consumed:
            del self.game.energy_orbs[oid]

        if orb_moved or orbs_consumed:
            self.game._energy_orbs_cache = None

    # ── Meteor Shower ──

    def _tick_meteor_shower(self, current_time: float):
        # Spawn new meteors at intervals
        if current_time - self.last_meteor_time >= METEOR_INTERVAL:
            self.last_meteor_time = current_time
            for _ in range(METEOR_COUNT_PER_WAVE):
                mx = random.uniform(50, WORLD_WIDTH - 50)
                my = random.uniform(50, WORLD_HEIGHT - 50)
                self.meteors.append(Meteor(x=mx, y=my, impact_time=current_time))

                # Damage players in blast radius
                for player in self.game.players.values():
                    if not player.alive:
                        continue
                    if player.check_invincible(current_time):
                        continue
                    has_shield = player.active_powerup == "shield" and current_time < player.powerup_until
                    if has_shield:
                        continue
                    # Check if player is sheltered by a wall
                    sheltered = False
                    for wall in self.game.walls.values():
                        if (wall.x <= player.x <= wall.x + wall.width and
                                wall.y <= player.y <= wall.y + wall.height):
                            sheltered = True
                            break
                        # Also check if near a wall (within player radius)
                        closest_x = max(wall.x, min(player.x, wall.x + wall.width))
                        closest_y = max(wall.y, min(player.y, wall.y + wall.height))
                        wall_dist = math.sqrt((player.x - closest_x)**2 + (player.y - closest_y)**2)
                        if wall_dist < player.radius * 0.5:
                            sheltered = True
                            break
                    if sheltered:
                        continue

                    dx = player.x - mx
                    dy = player.y - my
                    if dx * dx + dy * dy < (METEOR_BLAST_RADIUS + player.radius) ** 2:
                        player.radius = max(MIN_RADIUS, player.radius - METEOR_DAMAGE)
                        if player.radius <= MIN_RADIUS:
                            player.alive = False
                            player.score = 0
                            self.game.add_kill("Meteor", player.name)

        # Clean up old meteor markers (keep for 0.5s for visual)
        self.meteors = [m for m in self.meteors if current_time - m.impact_time < 0.5]

    # ── Feeding Frenzy ──

    def _spawn_frenzy_orbs(self):
        self.frenzy_orb_ids = []
        for _ in range(FRENZY_ORB_COUNT):
            self.game.orb_counter += 1
            orb_id = f"frenzy_{self.game.orb_counter}"
            x, y = self.game.find_safe_orb_position(ENERGY_ORB_RADIUS)
            hue = 0.25 + random.random() * 0.15
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            self.game.energy_orbs[orb_id] = EnergyOrb(id=orb_id, x=x, y=y, color=color)
            self.frenzy_orb_ids.append(orb_id)
        self.game._energy_orbs_cache = None

    # ── Supernova ──

    def _apply_supernova(self):
        loss_pct = random.uniform(SUPERNOVA_MASS_LOSS_MIN, SUPERNOVA_MASS_LOSS_MAX)
        for player in self.game.players.values():
            if not player.alive:
                continue
            dx = player.x - self.supernova_x
            dy = player.y - self.supernova_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < SUPERNOVA_RADIUS:
                player.radius = max(MIN_RADIUS, player.radius * (1 - loss_pct))
                player.score = max(0, int(player.score * (1 - loss_pct * 0.5)))

    # ── Earthquake ──

    def _generate_new_wall_positions(self) -> list:
        new_walls = []
        for old in self.earthquake_old_walls:
            # Randomize position but keep same dimensions
            new_x = random.uniform(100, WORLD_WIDTH - 100 - old["width"])
            new_y = random.uniform(100, WORLD_HEIGHT - 100 - old["height"])
            new_walls.append({
                "id": old["id"], "x": new_x, "y": new_y,
                "width": old["width"], "height": old["height"]
            })
        return new_walls

    def _tick_earthquake(self, current_time: float):
        elapsed = current_time - self.disaster_start
        self.earthquake_progress = min(1.0, elapsed / EARTHQUAKE_DURATION)
        t = self.earthquake_progress
        # Smooth easing
        t = t * t * (3 - 2 * t)

        for old, new in zip(self.earthquake_old_walls, self.earthquake_new_walls):
            wall = self.game.walls.get(old["id"])
            if wall:
                wall.x = old["x"] + (new["x"] - old["x"]) * t
                wall.y = old["y"] + (new["y"] - old["y"]) * t
        self.game._walls_cache = None

    def _finalize_earthquake(self):
        for new in self.earthquake_new_walls:
            wall = self.game.walls.get(new["id"])
            if wall:
                wall.x = new["x"]
                wall.y = new["y"]
        self.game._walls_cache = None

    def get_state(self, current_time: float) -> dict:
        """Return disaster state for broadcast to clients."""
        state = {
            "active": self.active_disaster,
            "warning": self.warning_type if self.warning_active else None,
            "warning_remaining": round(max(0, DISASTER_WARNING_TIME - (current_time - self.warning_start)), 1) if self.warning_active else 0,
        }
        if self.active_disaster:
            elapsed = current_time - self.disaster_start
            duration = self.disaster_end - self.disaster_start
            state["remaining"] = round(max(0, duration - elapsed), 1)
            state["progress"] = round(min(1.0, elapsed / duration), 2)

        if self.active_disaster == "black_hole" and self.black_hole:
            state["black_hole"] = self.black_hole.to_dict()
        elif self.active_disaster == "meteor_shower":
            state["meteors"] = [m.to_dict() for m in self.meteors]
        elif self.active_disaster == "fog_of_war":
            state["fog_radius"] = FOG_VISIBILITY_RADIUS
        elif self.active_disaster == "supernova":
            state["supernova"] = {
                "x": round(self.supernova_x, 1),
                "y": round(self.supernova_y, 1),
                "radius": SUPERNOVA_RADIUS,
                "time": round(current_time - self.supernova_time, 2)
            }
        elif self.active_disaster == "earthquake":
            state["earthquake_progress"] = round(self.earthquake_progress, 2)

        return state


class GameState:
    def __init__(self):
        self.players: Dict[str, Player] = {}
        self.energy_orbs: Dict[str, EnergyOrb] = {}
        self.spike_orbs: Dict[str, SpikeOrb] = {}
        self.golden_orbs: Dict[str, GoldenOrb] = {}
        self.walls: Dict[str, Wall] = {}
        self.projectiles: Dict[str, Projectile] = {}
        self.powerup_orbs: Dict[str, PowerUpOrb] = {}
        self.connections: Dict[str, any] = {}
        self.orb_counter = 0
        self.spike_counter = 0
        self.golden_counter = 0
        self.wall_counter = 0
        self.projectile_counter = 0
        self.powerup_counter = 0
        self.powerup_respawn_timers: list = []  # [(respawn_time), ...]
        # Kill feed
        self.kill_feed: list = []  # [(timestamp, killer_name, victim_name), ...]
        # Leaderboard cache (updated every 1 second instead of every tick)
        self._cached_leaderboard: list = []
        self._leaderboard_update_time: float = 0
        self._leaderboard_cache_duration: float = 1.0  # seconds
        # Orb serialization caches (invalidated on collect/respawn)
        self._energy_orbs_cache: list = None
        self._spike_orbs_cache: list = None
        self._golden_orbs_cache: list = None
        self._powerup_orbs_cache: list = None
        self._walls_cache: list = None
        self.spawn_walls()
        self.spawn_energy_orbs(ENERGY_ORB_COUNT)
        self.spawn_spike_orbs(SPIKE_ORB_COUNT)
        self.spawn_golden_orbs(GOLDEN_ORB_COUNT)
        self.spawn_powerup_orbs(POWERUP_COUNT)
        self.disaster_manager = DisasterManager(self)

    def generate_color(self) -> str:
        """Generate a vibrant random color."""
        hue = random.random()
        saturation = 0.7 + random.random() * 0.3
        value = 0.8 + random.random() * 0.2
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def find_safe_orb_position(self, radius: float = 10) -> tuple:
        """Find a random position not inside any wall."""
        for _ in range(50):
            x = random.uniform(50, WORLD_WIDTH - 50)
            y = random.uniform(50, WORLD_HEIGHT - 50)
            inside_wall = False
            for wall in self.walls.values():
                if (wall.x - radius < x < wall.x + wall.width + radius and
                    wall.y - radius < y < wall.y + wall.height + radius):
                    inside_wall = True
                    break
            if not inside_wall:
                return x, y
        return x, y  # fallback to last attempt

    def spawn_energy_orbs(self, count: int):
        """Spawn energy orbs at random positions."""
        for _ in range(count):
            self.orb_counter += 1
            orb_id = f"orb_{self.orb_counter}"
            # Random green-ish color
            hue = 0.25 + random.random() * 0.15  # Green range
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

            x, y = self.find_safe_orb_position(ENERGY_ORB_RADIUS)
            self.energy_orbs[orb_id] = EnergyOrb(
                id=orb_id,
                x=x,
                y=y,
                color=color
            )

    def spawn_spike_orbs(self, count: int):
        """Spawn evil spike orbs at random positions."""
        for _ in range(count):
            self.spike_counter += 1
            orb_id = f"spike_{self.spike_counter}"
            # Random red/pink color
            hue = random.uniform(0.95, 1.0) if random.random() > 0.5 else random.uniform(0.0, 0.05)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

            x, y = self.find_safe_orb_position(SPIKE_ORB_RADIUS)
            self.spike_orbs[orb_id] = SpikeOrb(
                id=orb_id,
                x=x,
                y=y,
                color=color
            )

    def spawn_golden_orbs(self, count: int):
        """Spawn rare golden orbs worth extra points."""
        for _ in range(count):
            self.golden_counter += 1
            orb_id = f"golden_{self.golden_counter}"
            x, y = self.find_safe_orb_position(GOLDEN_ORB_RADIUS)
            self.golden_orbs[orb_id] = GoldenOrb(
                id=orb_id,
                x=x,
                y=y
            )

    def spawn_powerup_orbs(self, count: int):
        """Spawn mystery power-up orbs."""
        for _ in range(count):
            self.powerup_counter += 1
            orb_id = f"powerup_{self.powerup_counter}"
            x, y = self.find_safe_orb_position(POWERUP_RADIUS)
            self.powerup_orbs[orb_id] = PowerUpOrb(
                id=orb_id,
                x=x,
                y=y
            )
        self._powerup_orbs_cache = None

    def spawn_walls(self):
        """Spawn obstacle walls around the map."""
        wall_configs = [
            # Central cross
            {"x": WORLD_WIDTH // 2 - 150, "y": WORLD_HEIGHT // 2 - 25, "width": 300, "height": 50},
            {"x": WORLD_WIDTH // 2 - 25, "y": WORLD_HEIGHT // 2 - 150, "width": 50, "height": 300},
            # Corner L-shapes
            {"x": 200, "y": 200, "width": 150, "height": 30},
            {"x": 200, "y": 200, "width": 30, "height": 150},
            {"x": WORLD_WIDTH - 350, "y": 200, "width": 150, "height": 30},
            {"x": WORLD_WIDTH - 230, "y": 200, "width": 30, "height": 150},
            {"x": 200, "y": WORLD_HEIGHT - 230, "width": 150, "height": 30},
            {"x": 200, "y": WORLD_HEIGHT - 350, "width": 30, "height": 150},
            {"x": WORLD_WIDTH - 350, "y": WORLD_HEIGHT - 230, "width": 150, "height": 30},
            {"x": WORLD_WIDTH - 230, "y": WORLD_HEIGHT - 350, "width": 30, "height": 150},
        ]
        for i, cfg in enumerate(wall_configs):
            wall_id = f"wall_{i}"
            self.walls[wall_id] = Wall(
                id=wall_id,
                x=cfg["x"],
                y=cfg["y"],
                width=cfg["width"],
                height=cfg["height"]
            )

    def add_kill(self, killer_name: str, victim_name: str):
        """Add a kill to the feed."""
        self.kill_feed.append({
            "time": time.time(),
            "killer": killer_name,
            "victim": victim_name
        })
        # Keep only recent kills
        if len(self.kill_feed) > KILL_FEED_MAX * 2:
            self.kill_feed = self.kill_feed[-KILL_FEED_MAX:]

    def get_kill_feed(self) -> list:
        """Get recent kills for display."""
        current_time = time.time()
        # Filter to recent kills only
        recent = [k for k in self.kill_feed if current_time - k["time"] < KILL_FEED_DURATION]
        return recent[-KILL_FEED_MAX:]

    def activate_boost(self, player_id: str):
        """Activate boost for a player."""
        if player_id not in self.players:
            return
        player = self.players[player_id]
        current_time = time.time()

        # Check cooldown and minimum size
        if current_time < player.boost_cooldown_until:
            return
        if player.radius <= MIN_RADIUS + BOOST_MASS_COST:
            return

        # Activate boost
        player.boost_active_until = current_time + BOOST_DURATION
        player.boost_cooldown_until = current_time + BOOST_COOLDOWN
        player.radius -= BOOST_MASS_COST

    def shoot(self, player_id: str, target_x: float, target_y: float):
        """Fire a projectile from a player toward a target position."""
        if player_id not in self.players:
            return
        player = self.players[player_id]
        current_time = time.time()

        if not player.alive:
            return
        if player.radius < PROJECTILE_MIN_RADIUS:
            return

        has_rapid_fire = player.active_powerup == "rapid_fire" and current_time < player.powerup_until

        if not has_rapid_fire and current_time < player.shoot_cooldown_until:
            return

        # Calculate direction
        dx = target_x - player.x
        dy = target_y - player.y
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 1:
            return

        # Normalize direction
        ndx = dx / distance
        ndy = dy / distance

        # Cost mass (free with rapid fire)
        if not has_rapid_fire:
            player.radius -= PROJECTILE_COST
            player.shoot_cooldown_until = current_time + PROJECTILE_COOLDOWN

        # Spawn projectile at player's edge
        self.projectile_counter += 1
        proj_id = f"proj_{self.projectile_counter}"
        self.projectiles[proj_id] = Projectile(
            id=proj_id,
            owner_id=player_id,
            x=player.x + ndx * (player.radius + PROJECTILE_RADIUS + 2),
            y=player.y + ndy * (player.radius + PROJECTILE_RADIUS + 2),
            dx=ndx,
            dy=ndy,
            color=player.color,
            created_at=current_time
        )

    def add_player(self, player_id: str, name: str, websocket) -> Player:
        """Add a new player to the game."""
        # Spawn at random position away from walls and other players
        x, y = self.find_safe_spawn()

        player = Player(
            id=player_id,
            name=sanitize_name(name),
            x=x,
            y=y,
            radius=INITIAL_RADIUS,
            color=self.generate_color(),
            target_x=x,
            target_y=y,
            invincible_until=time.time() + RESPAWN_INVINCIBILITY
        )
        self.players[player_id] = player
        self.connections[player_id] = websocket
        return player

    def find_safe_spawn(self) -> tuple:
        """Find a spawn point not inside a wall."""
        for _ in range(50):  # Max attempts
            x = random.uniform(100, WORLD_WIDTH - 100)
            y = random.uniform(100, WORLD_HEIGHT - 100)
            # Check if inside any wall
            safe = True
            for wall in self.walls.values():
                if (wall.x - 50 < x < wall.x + wall.width + 50 and
                    wall.y - 50 < y < wall.y + wall.height + 50):
                    safe = False
                    break
            if safe:
                return x, y
        # Fallback
        return WORLD_WIDTH // 2, WORLD_HEIGHT // 2

    def remove_player(self, player_id: str):
        """Remove a player from the game."""
        if player_id in self.players:
            del self.players[player_id]
        if player_id in self.connections:
            del self.connections[player_id]

    def update_player_target(self, player_id: str, target_x: float, target_y: float):
        """Update where a player is moving towards."""
        if player_id in self.players:
            self.players[player_id].target_x = target_x
            self.players[player_id].target_y = target_y

    def respawn_player(self, player_id: str):
        """Respawn a dead player."""
        if player_id in self.players:
            player = self.players[player_id]
            x, y = self.find_safe_spawn()
            player.x = x
            player.y = y
            player.radius = INITIAL_RADIUS
            player.target_x = player.x
            player.target_y = player.y
            player.alive = True
            player.invincible_until = time.time() + RESPAWN_INVINCIBILITY
            player.boost_cooldown_until = 0
            player.boost_active_until = 0
            player.shoot_cooldown_until = 0
            player.critical_mass_start = 0
            player.active_powerup = ""
            player.powerup_until = 0

    def tick(self):
        """Update game state for one tick."""
        current_time = time.time()

        # Move players towards their targets
        for player in self.players.values():
            if not player.alive:
                continue

            prev_x, prev_y = player.x, player.y

            dx = player.target_x - player.x
            dy = player.target_y - player.y
            dist_sq = dx * dx + dy * dy

            if dist_sq > 25:  # 5^2
                distance = math.sqrt(dist_sq)
                speed = player.get_speed(current_time)
                player.x += (dx / distance) * speed
                player.y += (dy / distance) * speed

            # Keep player in bounds
            player.x = max(player.radius, min(WORLD_WIDTH - player.radius, player.x))
            player.y = max(player.radius, min(WORLD_HEIGHT - player.radius, player.y))

            # Wall collisions - push player out of walls (phantom passes through)
            wall_iters = 0 if (player.active_powerup == "phantom" and current_time < player.powerup_until) else 3
            for _iteration in range(wall_iters):
                for wall in self.walls.values():
                    # Check if player overlaps with wall
                    closest_x = max(wall.x, min(player.x, wall.x + wall.width))
                    closest_y = max(wall.y, min(player.y, wall.y + wall.height))
                    dist_x = player.x - closest_x
                    dist_y = player.y - closest_y
                    dist = math.sqrt(dist_x * dist_x + dist_y * dist_y)

                    if dist < player.radius:
                        if dist > 0:
                            overlap = player.radius - dist + 1
                            player.x += (dist_x / dist) * overlap
                            player.y += (dist_y / dist) * overlap
                        else:
                            # Player center is inside wall - push back toward previous position
                            push_dx = prev_x - player.x
                            push_dy = prev_y - player.y
                            push_dist = math.sqrt(push_dx * push_dx + push_dy * push_dy)
                            if push_dist > 0:
                                # Push back the way they came
                                player.x = prev_x
                                player.y = prev_y
                            else:
                                # Fallback: push to nearest edge
                                push_left = player.x - wall.x
                                push_right = (wall.x + wall.width) - player.x
                                push_up = player.y - wall.y
                                push_down = (wall.y + wall.height) - player.y
                                min_push = min(push_left, push_right, push_up, push_down)
                                if min_push == push_left:
                                    player.x = wall.x - player.radius - 1
                                elif min_push == push_right:
                                    player.x = wall.x + wall.width + player.radius + 1
                                elif min_push == push_up:
                                    player.y = wall.y - player.radius - 1
                                else:
                                    player.y = wall.y + wall.height + player.radius + 1

            # Slowly shrink (but not below minimum)
            if player.radius > MIN_RADIUS + 5:
                player.radius = max(MIN_RADIUS, player.radius - SHRINK_RATE)

        # Check energy orb collisions
        orbs_to_remove = []
        for orb_id, orb in self.energy_orbs.items():
            for player in self.players.values():
                if not player.alive:
                    continue
                dx = player.x - orb.x
                dy = player.y - orb.y
                combined = player.radius + orb.radius

                if dx * dx + dy * dy < combined * combined:
                    player.radius = min(MAX_RADIUS, player.radius + ENERGY_ORB_VALUE)
                    player.score += 10
                    orbs_to_remove.append(orb_id)
                    break

        # Remove collected orbs and spawn new ones
        if orbs_to_remove:
            for orb_id in orbs_to_remove:
                del self.energy_orbs[orb_id]
            self.spawn_energy_orbs(len(orbs_to_remove))
            self._energy_orbs_cache = None

        # Check spike orb collisions (evil orbs that halve your size!)
        spikes_to_remove = []
        for orb_id, orb in self.spike_orbs.items():
            for player in self.players.values():
                if not player.alive:
                    continue
                # Shield and phantom block spike damage
                if player.active_powerup in ("shield", "phantom") and current_time < player.powerup_until:
                    continue
                dx = player.x - orb.x
                dy = player.y - orb.y
                combined = player.radius + orb.radius

                if dx * dx + dy * dy < combined * combined:
                    player.radius = max(MIN_RADIUS, player.radius * 0.5)
                    player.score = max(0, player.score // 2)
                    spikes_to_remove.append(orb_id)
                    break

        # Remove collected spikes and spawn new ones
        if spikes_to_remove:
            for orb_id in spikes_to_remove:
                del self.spike_orbs[orb_id]
            self.spawn_spike_orbs(len(spikes_to_remove))
            self._spike_orbs_cache = None

        # Check golden orb collisions (rare, high value!)
        golden_to_remove = []
        for orb_id, orb in self.golden_orbs.items():
            for player in self.players.values():
                if not player.alive:
                    continue
                dx = player.x - orb.x
                dy = player.y - orb.y
                combined = player.radius + orb.radius

                if dx * dx + dy * dy < combined * combined:
                    player.radius = min(MAX_RADIUS, player.radius + GOLDEN_ORB_VALUE)
                    player.score += 50
                    golden_to_remove.append(orb_id)
                    break

        # Remove collected golden orbs and spawn new ones
        if golden_to_remove:
            for orb_id in golden_to_remove:
                del self.golden_orbs[orb_id]
            self.spawn_golden_orbs(len(golden_to_remove))
            self._golden_orbs_cache = None

        # Check power-up orb collisions
        powerups_to_remove = []
        for orb_id, orb in self.powerup_orbs.items():
            for player in self.players.values():
                if not player.alive:
                    continue
                dx = player.x - orb.x
                dy = player.y - orb.y
                combined = player.radius + orb.radius
                if dx * dx + dy * dy < combined * combined:
                    powerup_type = random.choice(POWERUP_TYPES)
                    player.active_powerup = powerup_type
                    player.powerup_until = current_time + POWERUP_DURATIONS[powerup_type]
                    powerups_to_remove.append(orb_id)
                    break

        if powerups_to_remove:
            for orb_id in powerups_to_remove:
                del self.powerup_orbs[orb_id]
            self._powerup_orbs_cache = None
            for _ in powerups_to_remove:
                self.powerup_respawn_timers.append(current_time + POWERUP_RESPAWN_DELAY)

        # Power-up respawn timer
        if self.powerup_respawn_timers:
            respawns = [t for t in self.powerup_respawn_timers if current_time >= t]
            if respawns:
                self.powerup_respawn_timers = [t for t in self.powerup_respawn_timers if current_time < t]
                self.spawn_powerup_orbs(len(respawns))

        # Check player vs player collisions
        players_list = list(self.players.values())
        for i, player1 in enumerate(players_list):
            if not player1.alive:
                continue
            for player2 in players_list[i+1:]:
                if not player2.alive:
                    continue

                # Skip if either player is invincible, shielded, or phantom
                if (player1.check_invincible(current_time) or player2.check_invincible(current_time) or
                    (player1.active_powerup in ("shield", "phantom") and current_time < player1.powerup_until) or
                    (player2.active_powerup in ("shield", "phantom") and current_time < player2.powerup_until)):
                    continue

                dx = player1.x - player2.x
                dy = player1.y - player2.y
                combined = player1.radius + player2.radius
                dist_sq = dx * dx + dy * dy

                if dist_sq < combined * combined:
                    distance = math.sqrt(dist_sq)
                    # Collision! Larger player consumes smaller
                    if player1.radius > player2.radius * CONSUME_RATIO:
                        # Player 1 consumes Player 2
                        player1.radius = min(MAX_RADIUS, player1.radius + player2.radius * 0.5)
                        player1.score += 100 + int(player2.score * 0.1)
                        player2.alive = False
                        player2.score = 0
                        self.add_kill(player1.name, player2.name)
                    elif player2.radius > player1.radius * CONSUME_RATIO:
                        # Player 2 consumes Player 1
                        player2.radius = min(MAX_RADIUS, player2.radius + player1.radius * 0.5)
                        player2.score += 100 + int(player1.score * 0.1)
                        player1.alive = False
                        player1.score = 0
                        self.add_kill(player2.name, player1.name)
                    else:
                        # Similar size - bounce off each other
                        if distance > 0:
                            overlap = (player1.radius + player2.radius - distance) / 2
                            player1.x += (dx / distance) * overlap
                            player1.y += (dy / distance) * overlap
                            player2.x -= (dx / distance) * overlap
                            player2.y -= (dy / distance) * overlap

        # Update projectiles
        projectiles_to_remove = []
        for proj_id, proj in self.projectiles.items():
            # Move projectile
            proj.x += proj.dx * PROJECTILE_SPEED
            proj.y += proj.dy * PROJECTILE_SPEED

            # Remove if expired or out of bounds
            if current_time - proj.created_at > PROJECTILE_LIFETIME:
                projectiles_to_remove.append(proj_id)
                continue
            if proj.x < 0 or proj.x > WORLD_WIDTH or proj.y < 0 or proj.y > WORLD_HEIGHT:
                projectiles_to_remove.append(proj_id)
                continue

            # Wall collisions
            hit_wall = False
            for wall in self.walls.values():
                closest_x = max(wall.x, min(proj.x, wall.x + wall.width))
                closest_y = max(wall.y, min(proj.y, wall.y + wall.height))
                dist_x = proj.x - closest_x
                dist_y = proj.y - closest_y
                dist = math.sqrt(dist_x * dist_x + dist_y * dist_y)
                if dist < proj.radius:
                    projectiles_to_remove.append(proj_id)
                    hit_wall = True
                    break
            if hit_wall:
                continue

            # Player collisions (skip owner and invincible players)
            for player in self.players.values():
                if (not player.alive or player.id == proj.owner_id or player.check_invincible(current_time) or
                    (player.active_powerup in ("shield", "phantom") and current_time < player.powerup_until)):
                    continue
                dx = player.x - proj.x
                dy = player.y - proj.y
                combined = player.radius + proj.radius
                if dx * dx + dy * dy < combined * combined:
                    # Hit! Reduce target radius
                    player.radius = max(MIN_RADIUS, player.radius - PROJECTILE_DAMAGE)
                    projectiles_to_remove.append(proj_id)
                    # Kill if target is at minimum and shooter exists and is larger
                    if player.radius <= MIN_RADIUS:
                        shooter = self.players.get(proj.owner_id)
                        if shooter and shooter.alive and shooter.radius > player.radius * CONSUME_RATIO:
                            player.alive = False
                            shooter.score += 100 + int(player.score * 0.1)
                            player.score = 0
                            self.add_kill(shooter.name, player.name)
                    break

        for proj_id in projectiles_to_remove:
            if proj_id in self.projectiles:
                del self.projectiles[proj_id]

        # Critical mass timer
        for player in self.players.values():
            if not player.alive:
                continue
            if player.radius >= CRITICAL_MASS_THRESHOLD:
                if player.critical_mass_start == 0:
                    # Start the timer
                    player.critical_mass_start = current_time
                elif current_time - player.critical_mass_start >= CRITICAL_MASS_TIMER:
                    # EXPLODE!
                    player.radius = INITIAL_RADIUS
                    player.score = max(0, player.score // 2)
                    player.critical_mass_start = 0
                    self.add_kill(player.name, f"{player.name} (exploded)")
            else:
                # Below threshold, reset timer
                player.critical_mass_start = 0

        # Power-up expiry
        for player in self.players.values():
            if player.active_powerup and current_time >= player.powerup_until:
                player.active_powerup = ""
                player.powerup_until = 0

        # Magnet effect - pull energy orbs toward magnet players
        magnet_moved = False
        for player in self.players.values():
            if not player.alive or player.active_powerup != "magnet" or current_time >= player.powerup_until:
                continue
            for orb in self.energy_orbs.values():
                dx = player.x - orb.x
                dy = player.y - orb.y
                dist_sq = dx * dx + dy * dy
                if dist_sq < MAGNET_RANGE * MAGNET_RANGE and dist_sq > 1:
                    dist = math.sqrt(dist_sq)
                    orb.x += (dx / dist) * MAGNET_STRENGTH
                    orb.y += (dy / dist) * MAGNET_STRENGTH
                    magnet_moved = True
        if magnet_moved:
            self._energy_orbs_cache = None

        # Natural disasters
        self.disaster_manager.tick(current_time)

    def get_static_data(self) -> dict:
        """Get static data that only needs to be sent once (on welcome)."""
        if self._walls_cache is None:
            self._walls_cache = [w.to_dict() for w in self.walls.values()]
        return {
            "walls": self._walls_cache,
            "world": {"width": WORLD_WIDTH, "height": WORLD_HEIGHT}
        }

    def build_shared_state(self, current_time: float) -> dict:
        """Build the shared portion of game state (called once per tick)."""
        # Use cached orb lists when available
        if self._energy_orbs_cache is None:
            self._energy_orbs_cache = [o.to_dict() for o in self.energy_orbs.values()]
        if self._spike_orbs_cache is None:
            self._spike_orbs_cache = [o.to_dict() for o in self.spike_orbs.values()]
        if self._golden_orbs_cache is None:
            self._golden_orbs_cache = [o.to_dict() for o in self.golden_orbs.values()]
        if self._powerup_orbs_cache is None:
            self._powerup_orbs_cache = [o.to_dict() for o in self.powerup_orbs.values()]

        return {
            "type": "state",
            "players": [p.to_dict(current_time) for p in self.players.values()],
            "energy_orbs": self._energy_orbs_cache,
            "spike_orbs": self._spike_orbs_cache,
            "golden_orbs": self._golden_orbs_cache,
            "powerup_orbs": self._powerup_orbs_cache,
            "projectiles": [p.to_dict() for p in self.projectiles.values()],
            "kill_feed": self.get_kill_feed(),
            "leaderboard": self.get_leaderboard(),
            "disaster": self.disaster_manager.get_state(current_time)
        }

    def get_leaderboard(self) -> list:
        """Get top 10 players by score (cached for performance)."""
        current_time = time.time()
        if current_time - self._leaderboard_update_time >= self._leaderboard_cache_duration:
            sorted_players = sorted(
                [p for p in self.players.values() if p.alive],
                key=lambda p: p.score,
                reverse=True
            )[:10]
            self._cached_leaderboard = [{"name": p.name, "score": p.score} for p in sorted_players]
            self._leaderboard_update_time = current_time
        return self._cached_leaderboard


# Global game state
game = GameState()


SEND_TIMEOUT = 0.5  # seconds - drop slow clients to prevent buffer buildup

# Rate limiting / connection cap
MAX_CONNECTIONS = 50
RATE_LIMIT_WINDOW = 1.0   # seconds
RATE_LIMIT_MAX_MSGS = 120  # max messages per window (30fps move + up to 60 shoots during rapid_fire)
active_connections = 0


async def broadcast_state():
    """Broadcast game state to all connected players."""
    while True:
        game.tick()
        current_time = time.time()

        # Build shared state once and serialize to JSON once
        shared_state = game.build_shared_state(current_time)
        # Serialize without 'you' - we'll splice it in per player
        shared_json = json.dumps(shared_state)
        # Remove trailing '}' so we can append ',"you":...}'
        shared_json_prefix = shared_json[:-1] + ',"you":'

        # Send state to each player (only serialize their 'you' portion)
        disconnected = []
        for player_id, websocket in list(game.connections.items()):
            player = game.players.get(player_id)
            if not player:
                continue
            try:
                you_json = json.dumps(player.to_dict(current_time))
                message = shared_json_prefix + you_json + '}'
                await asyncio.wait_for(
                    websocket.send(message),
                    timeout=SEND_TIMEOUT
                )
            except asyncio.TimeoutError:
                print(f"Player {player_id} send timeout - dropping connection")
                disconnected.append(player_id)
            except ConnectionClosed:
                disconnected.append(player_id)
            except Exception as e:
                print(f"Error sending to {player_id}: {e}")
                disconnected.append(player_id)

        # Clean up disconnected players
        for player_id in disconnected:
            game.remove_player(player_id)
            print(f"Player {player_id} disconnected")

        await asyncio.sleep(TICK_RATE)


async def handle_client(websocket):
    """Handle a single client connection."""
    global active_connections
    player_id = None

    # Enforce connection cap
    if active_connections >= MAX_CONNECTIONS:
        await websocket.close(1013, "Server full")
        return

    active_connections += 1
    # Rate limiting state for this client
    msg_count = 0
    window_start = time.time()

    try:
        # Wait for join message
        message = await websocket.recv()
        data = json.loads(message)

        if data.get("type") == "join":
            player_id = f"player_{id(websocket)}"
            name = sanitize_name(str(data.get("name", "Anonymous")))
            player = game.add_player(player_id, name, websocket)

            # Send welcome message with static data
            welcome_data = {
                "type": "welcome",
                "player_id": player_id,
                "player": player.to_dict(time.time())
            }
            welcome_data.update(game.get_static_data())
            await websocket.send(json.dumps(welcome_data))

            print(f"Player {name} ({player_id}) joined!")

            # Handle messages from this client
            async for message in websocket:
                # Rate limiting
                now = time.time()
                if now - window_start >= RATE_LIMIT_WINDOW:
                    msg_count = 0
                    window_start = now
                msg_count += 1
                if msg_count > RATE_LIMIT_MAX_MSGS:
                    continue  # Silently drop excess messages

                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "move":
                        game.update_player_target(
                            player_id,
                            safe_float(data.get("x", 0)),
                            safe_float(data.get("y", 0))
                        )

                    elif msg_type == "boost":
                        game.activate_boost(player_id)

                    elif msg_type == "shoot":
                        game.shoot(
                            player_id,
                            safe_float(data.get("x", 0)),
                            safe_float(data.get("y", 0))
                        )

                    elif msg_type == "respawn":
                        game.respawn_player(player_id)

                except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                    pass  # Silently drop malformed messages

    except ConnectionClosed:
        pass
    finally:
        active_connections -= 1
        if player_id:
            game.remove_player(player_id)
            print(f"Player {player_id} left")


def get_local_ip():
    """Get the local IP address for LAN play."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


ALLOWED_HTTP_FILES = {"/", "/index.html"}


class SafeHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that only serves the game client file."""

    def do_GET(self):
        # Normalize path and only allow index.html
        path = self.path.split("?")[0].split("#")[0]  # strip query/fragment
        if path not in ALLOWED_HTTP_FILES:
            self.send_error(404, "Not Found")
            return
        # Always serve index.html
        self.path = "/index.html"
        super().do_GET()

    def do_HEAD(self):
        path = self.path.split("?")[0].split("#")[0]
        if path not in ALLOWED_HTTP_FILES:
            self.send_error(404, "Not Found")
            return
        self.path = "/index.html"
        super().do_HEAD()

    def log_message(self, format, *args):
        pass  # Suppress routine request logs

    def log_error(self, format, *args):
        # Keep error logging for visibility into abuse attempts
        print(f"[HTTP] {self.client_address[0]} - {format % args}")


def start_http_server(port=8080):
    """Start a threaded HTTP server to serve the game files."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Use ThreadingHTTPServer to handle multiple connections
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), SafeHTTPHandler)
    httpd.serve_forever()


async def main():
    """Start the game server."""
    local_ip = get_local_ip()

    # Start HTTP server in a background thread
    http_thread = threading.Thread(target=start_http_server, args=(8080,), daemon=True)
    http_thread.start()

    print("=" * 50)
    print("  ORB ARENA - Multiplayer Game Server")
    print("=" * 50)
    print(f"  World Size: {WORLD_WIDTH}x{WORLD_HEIGHT}")
    print(f"  Tick Rate: {int(1/TICK_RATE)} FPS")
    print("=" * 50)
    print(f"\n  PLAY THE GAME:")
    print(f"    Local:  http://localhost:8080")
    print(f"    LAN:    http://{local_ip}:8080")
    print("=" * 50)
    print(f"\n  Share this URL with friends: http://{local_ip}:8080")
    print("\n  Press Ctrl+C to stop the server\n")

    # Start the game loop
    asyncio.create_task(broadcast_state())

    # Allowed origins for WebSocket connections (prevents cross-site hijacking)
    # None = allow all origins (for LAN play without domain setup)
    # Set ALLOWED_ORIGINS env var to restrict in production, e.g. "https://game.yourdomain.com"
    allowed_origins_env = os.environ.get("ALLOWED_ORIGINS")
    if allowed_origins_env:
        ws_origins = [o.strip() for o in allowed_origins_env.split(",")]
        print(f"  WebSocket origins restricted to: {ws_origins}")
    else:
        ws_origins = None  # Allow all for LAN play
        print("  WebSocket origins: unrestricted (set ALLOWED_ORIGINS to restrict)")

    # Start WebSocket server (0.0.0.0 allows LAN connections)
    # Enable permessage-deflate compression to reduce bandwidth
    async with websockets.serve(
        handle_client, "0.0.0.0", 8765,
        compression="deflate",
        origins=ws_origins,
        max_size=1024,  # Max message size: 1KB (game messages are tiny)
    ):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")
