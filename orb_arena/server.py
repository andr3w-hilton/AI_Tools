"""
Orb Arena - Multiplayer WebSocket Game Server
A competitive arena game where players control orbs, collect energy, and consume smaller players.
"""

import asyncio
import json
import random
import math
import time
import websockets
from websockets.exceptions import ConnectionClosed
from dataclasses import dataclass, asdict
from typing import Dict, Set, Optional
import colorsys
import socket
import http.server
import threading
import os

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

# Respawn invincibility
RESPAWN_INVINCIBILITY = 3.0  # seconds

# Kill feed
KILL_FEED_MAX = 5  # max messages to show
KILL_FEED_DURATION = 5.0  # seconds before message expires

# Walls/Obstacles
WALL_COUNT = 8


@dataclass
class EnergyOrb:
    id: str
    x: float
    y: float
    radius: float = ENERGY_ORB_RADIUS
    color: str = "#00ff88"
    orb_type: str = "energy"

    def to_dict(self):
        return asdict(self)


@dataclass
class SpikeOrb:
    id: str
    x: float
    y: float
    radius: float = SPIKE_ORB_RADIUS
    color: str = "#ff2266"
    orb_type: str = "spike"

    def to_dict(self):
        return asdict(self)


@dataclass
class GoldenOrb:
    id: str
    x: float
    y: float
    radius: float = GOLDEN_ORB_RADIUS
    color: str = "#ffd700"
    orb_type: str = "golden"

    def to_dict(self):
        return asdict(self)


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

    def to_dict(self):
        current_time = time.time()
        return {
            "id": self.id,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
            "color": self.color,
            "score": self.score,
            "alive": self.alive,
            "is_boosting": current_time < self.boost_active_until,
            "boost_ready": current_time >= self.boost_cooldown_until,
            "is_invincible": current_time < self.invincible_until
        }

    @property
    def speed(self):
        # Larger players move slower, smaller players are much faster
        base = BASE_SPEED * (INITIAL_RADIUS / self.radius) ** SPEED_SCALING
        # Apply boost multiplier if active
        if time.time() < self.boost_active_until:
            return base * BOOST_SPEED_MULTIPLIER
        return base

    @property
    def is_invincible(self):
        return time.time() < self.invincible_until


class GameState:
    def __init__(self):
        self.players: Dict[str, Player] = {}
        self.energy_orbs: Dict[str, EnergyOrb] = {}
        self.spike_orbs: Dict[str, SpikeOrb] = {}
        self.golden_orbs: Dict[str, GoldenOrb] = {}
        self.walls: Dict[str, Wall] = {}
        self.connections: Dict[str, any] = {}
        self.orb_counter = 0
        self.spike_counter = 0
        self.golden_counter = 0
        self.wall_counter = 0
        # Kill feed
        self.kill_feed: list = []  # [(timestamp, killer_name, victim_name), ...]
        # Leaderboard cache (updated every 1 second instead of every tick)
        self._cached_leaderboard: list = []
        self._leaderboard_update_time: float = 0
        self._leaderboard_cache_duration: float = 1.0  # seconds
        self.spawn_energy_orbs(ENERGY_ORB_COUNT)
        self.spawn_spike_orbs(SPIKE_ORB_COUNT)
        self.spawn_golden_orbs(GOLDEN_ORB_COUNT)
        self.spawn_walls()

    def generate_color(self) -> str:
        """Generate a vibrant random color."""
        hue = random.random()
        saturation = 0.7 + random.random() * 0.3
        value = 0.8 + random.random() * 0.2
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def spawn_energy_orbs(self, count: int):
        """Spawn energy orbs at random positions."""
        for _ in range(count):
            self.orb_counter += 1
            orb_id = f"orb_{self.orb_counter}"
            # Random green-ish color
            hue = 0.25 + random.random() * 0.15  # Green range
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

            self.energy_orbs[orb_id] = EnergyOrb(
                id=orb_id,
                x=random.uniform(50, WORLD_WIDTH - 50),
                y=random.uniform(50, WORLD_HEIGHT - 50),
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

            self.spike_orbs[orb_id] = SpikeOrb(
                id=orb_id,
                x=random.uniform(50, WORLD_WIDTH - 50),
                y=random.uniform(50, WORLD_HEIGHT - 50),
                color=color
            )

    def spawn_golden_orbs(self, count: int):
        """Spawn rare golden orbs worth extra points."""
        for _ in range(count):
            self.golden_counter += 1
            orb_id = f"golden_{self.golden_counter}"
            self.golden_orbs[orb_id] = GoldenOrb(
                id=orb_id,
                x=random.uniform(50, WORLD_WIDTH - 50),
                y=random.uniform(50, WORLD_HEIGHT - 50)
            )

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

    def add_player(self, player_id: str, name: str, websocket) -> Player:
        """Add a new player to the game."""
        # Spawn at random position away from walls and other players
        x, y = self.find_safe_spawn()

        player = Player(
            id=player_id,
            name=name[:15],  # Limit name length
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

    def tick(self):
        """Update game state for one tick."""
        # Move players towards their targets
        for player in self.players.values():
            if not player.alive:
                continue

            dx = player.target_x - player.x
            dy = player.target_y - player.y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance > 5:
                # Normalize and apply speed
                player.x += (dx / distance) * player.speed
                player.y += (dy / distance) * player.speed

            # Keep player in bounds
            player.x = max(player.radius, min(WORLD_WIDTH - player.radius, player.x))
            player.y = max(player.radius, min(WORLD_HEIGHT - player.radius, player.y))

            # Wall collisions - push player out of walls
            for wall in self.walls.values():
                # Check if player overlaps with wall
                closest_x = max(wall.x, min(player.x, wall.x + wall.width))
                closest_y = max(wall.y, min(player.y, wall.y + wall.height))
                dist_x = player.x - closest_x
                dist_y = player.y - closest_y
                dist = math.sqrt(dist_x * dist_x + dist_y * dist_y)

                if dist < player.radius:
                    # Push player out
                    if dist > 0:
                        overlap = player.radius - dist
                        player.x += (dist_x / dist) * overlap
                        player.y += (dist_y / dist) * overlap
                    else:
                        # Player center is inside wall, push to nearest edge
                        player.x = wall.x - player.radius - 1

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
                distance = math.sqrt(dx * dx + dy * dy)

                if distance < player.radius + orb.radius:
                    # Player collected the orb
                    player.radius = min(MAX_RADIUS, player.radius + ENERGY_ORB_VALUE)
                    player.score += 10
                    orbs_to_remove.append(orb_id)
                    break

        # Remove collected orbs and spawn new ones
        for orb_id in orbs_to_remove:
            del self.energy_orbs[orb_id]
        if orbs_to_remove:
            self.spawn_energy_orbs(len(orbs_to_remove))

        # Check spike orb collisions (evil orbs that halve your size!)
        spikes_to_remove = []
        for orb_id, orb in self.spike_orbs.items():
            for player in self.players.values():
                if not player.alive:
                    continue
                dx = player.x - orb.x
                dy = player.y - orb.y
                distance = math.sqrt(dx * dx + dy * dy)

                if distance < player.radius + orb.radius:
                    # Player hit a spike - halve their size and score!
                    player.radius = max(MIN_RADIUS, player.radius * 0.5)
                    player.score = max(0, player.score // 2)
                    spikes_to_remove.append(orb_id)
                    break

        # Remove collected spikes and spawn new ones
        for orb_id in spikes_to_remove:
            del self.spike_orbs[orb_id]
        if spikes_to_remove:
            self.spawn_spike_orbs(len(spikes_to_remove))

        # Check golden orb collisions (rare, high value!)
        golden_to_remove = []
        for orb_id, orb in self.golden_orbs.items():
            for player in self.players.values():
                if not player.alive:
                    continue
                dx = player.x - orb.x
                dy = player.y - orb.y
                distance = math.sqrt(dx * dx + dy * dy)

                if distance < player.radius + orb.radius:
                    # Player collected the golden orb
                    player.radius = min(MAX_RADIUS, player.radius + GOLDEN_ORB_VALUE)
                    player.score += 50
                    golden_to_remove.append(orb_id)
                    break

        # Remove collected golden orbs and spawn new ones
        for orb_id in golden_to_remove:
            del self.golden_orbs[orb_id]
        if golden_to_remove:
            self.spawn_golden_orbs(len(golden_to_remove))

        # Check player vs player collisions
        players_list = list(self.players.values())
        for i, player1 in enumerate(players_list):
            if not player1.alive:
                continue
            for player2 in players_list[i+1:]:
                if not player2.alive:
                    continue

                # Skip if either player is invincible
                if player1.is_invincible or player2.is_invincible:
                    continue

                dx = player1.x - player2.x
                dy = player1.y - player2.y
                distance = math.sqrt(dx * dx + dy * dy)

                if distance < player1.radius + player2.radius:
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

    def get_state_for_player(self, player_id: str) -> dict:
        """Get the game state from a player's perspective."""
        player = self.players.get(player_id)
        if not player:
            return {}

        return {
            "type": "state",
            "you": player.to_dict(),
            "players": [p.to_dict() for p in self.players.values()],
            "energy_orbs": [o.to_dict() for o in self.energy_orbs.values()],
            "spike_orbs": [o.to_dict() for o in self.spike_orbs.values()],
            "golden_orbs": [o.to_dict() for o in self.golden_orbs.values()],
            "walls": [w.to_dict() for w in self.walls.values()],
            "kill_feed": self.get_kill_feed(),
            "world": {
                "width": WORLD_WIDTH,
                "height": WORLD_HEIGHT
            },
            "leaderboard": self.get_leaderboard()
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


async def broadcast_state():
    """Broadcast game state to all connected players."""
    while True:
        game.tick()

        # Send state to each player
        disconnected = []
        for player_id, websocket in list(game.connections.items()):
            try:
                state = game.get_state_for_player(player_id)
                if state:
                    # Timeout prevents memory buildup from slow clients
                    await asyncio.wait_for(
                        websocket.send(json.dumps(state)),
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
    player_id = None

    try:
        # Wait for join message
        message = await websocket.recv()
        data = json.loads(message)

        if data.get("type") == "join":
            player_id = f"player_{id(websocket)}"
            name = data.get("name", "Anonymous")[:15]
            player = game.add_player(player_id, name, websocket)

            # Send welcome message
            await websocket.send(json.dumps({
                "type": "welcome",
                "player_id": player_id,
                "player": player.to_dict()
            }))

            print(f"Player {name} ({player_id}) joined!")

            # Handle messages from this client
            async for message in websocket:
                try:
                    data = json.loads(message)

                    if data.get("type") == "move":
                        # Update player's target position
                        game.update_player_target(
                            player_id,
                            data.get("x", 0),
                            data.get("y", 0)
                        )

                    elif data.get("type") == "boost":
                        game.activate_boost(player_id)

                    elif data.get("type") == "respawn":
                        game.respawn_player(player_id)

                except json.JSONDecodeError:
                    pass

    except ConnectionClosed:
        pass
    finally:
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


class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that doesn't spam the console."""
    def log_message(self, format, *args):
        pass  # Suppress HTTP request logs


def start_http_server(port=8080):
    """Start a threaded HTTP server to serve the game files."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Use ThreadingHTTPServer to handle multiple connections
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), QuietHTTPHandler)
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

    # Start WebSocket server (0.0.0.0 allows LAN connections)
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")
