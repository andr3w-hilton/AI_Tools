lua_code = r"""-- dystopian metroidvania - day 5: dash + gun

local grav=0.3
local mxfall=4
local spd=1.5
local gfric=0.75
local afric=0.9
local jforce=-4.0
local coymax=8
local jbfmax=8
local tspd=8
local atk_dur=8
local dash_spd=4
local dash_dur=10
local dash_cd_max=25
local gun_cd_max=15

p={
 x=24,y=80,vx=0,vy=0,
 gnd=false,coy=0,jbf=0,
 dir=1,hp=4,
 djump=true,djumped=false,
 atk=false,atk_t=0,iframe=0,
 dashing=false,dash_t=0,dash_cd=0,
 weapon=0,
 gun=true,
 gun_cd=0
}

enemies={}
bullets={}
cam={x=0,y=0,tx=0,ty=0}
state="play"

function make_enemy(x,y)
 return {x=x,y=y,vx=-0.6,hp=2,hurt=0,hit=false}
end

function _init()
 add(enemies,make_enemy(160,96))
 add(enemies,make_enemy(210,96))
 add(enemies,make_enemy(310,96))
 add(enemies,make_enemy(530,96))
 add(enemies,make_enemy(590,96))
end

function _update()
 if state=="play" then
  pmove()
  player_atk()
  update_bullets()
  update_enemies()
  check_exit()
 else
  do_trans()
 end
end

function _draw()
 cls(1)
 camera(cam.x,cam.y)
 map(cam.x\8,cam.y\8,cam.x,cam.y,32,32)

 for b in all(bullets) do
  rectfill(b.x,b.y+1,b.x+5,b.y+3,10)
 end

 for e in all(enemies) do
  if e.hurt>0 then pal(9,7) end
  spr(4,e.x,e.y)
  pal()
 end

 if p.atk then
  local hx=p.dir==1 and p.x+8 or p.x-10
  rectfill(hx,p.y+2,hx+9,p.y+5,10)
 end

 if p.iframe==0 or p.iframe%4<2 then
  if p.dashing then pal(7,12) end
  spr(3,p.x,p.y,1,1,p.dir==-1)
  pal()
 end

 camera()
 for i=1,4 do
  circfill(i*8-2,6,2,i<=p.hp and 8 or 5)
 end
 local wc=p.weapon==0 and 6 or 10
 print(p.weapon==0 and "bld" or "gun",2,14,wc)
 local dp=1-(p.dash_cd/dash_cd_max)
 rectfill(2,23,2+flr(dp*20),24,p.dash_cd==0 and 12 or 5)
 print((cam.x\128)..",".. (cam.y\128),110,120,5)
end

function player_atk()
 if btnp(3) and p.gun then
  p.weapon=1-p.weapon
 end
 if btnp(5) then
  if p.weapon==0 then
   if not p.atk then
    p.atk=true p.atk_t=atk_dur
    for e in all(enemies) do e.hit=false end
   end
  elseif p.gun_cd==0 then
   local bx=p.dir==1 and p.x+8 or p.x-6
   add(bullets,{x=bx,y=p.y+3,vx=p.dir*5})
   p.gun_cd=gun_cd_max
  end
 end
 if p.gun_cd>0 then p.gun_cd-=1 end
 if p.atk then
  p.atk_t-=1
  if p.atk_t<=0 then p.atk=false end
  local hx=p.dir==1 and p.x+8 or p.x-10
  for e in all(enemies) do
   if not e.hit and overlap(hx,p.y+1,10,7,e.x,e.y,8,8) then
    e.hit=true e.hurt=8
    e.hp-=1
    if e.hp<=0 then del(enemies,e) end
   end
  end
 end
end

function update_bullets()
 for b in all(bullets) do
  b.x+=b.vx
  if solid(b.x,b.y+2) or b.x<cam.x-8 or b.x>cam.x+136 then
   del(bullets,b)
  else
   for e in all(enemies) do
    if overlap(b.x,b.y,6,4,e.x,e.y,8,8) then
     e.hurt=8 e.hp-=1
     if e.hp<=0 then del(enemies,e) end
     del(bullets,b) break
    end
   end
  end
 end
end

function update_enemies()
 for e in all(enemies) do
  if e.hurt>0 then e.hurt-=1 end
  e.x+=e.vx
  if e.vx>0 and (solid(e.x+8,e.y+4) or not solid(e.x+8,e.y+9)) then
   e.vx=-0.6 e.x=(e.x+8)\8*8-8
  elseif e.vx<0 and (solid(e.x-1,e.y+4) or not solid(e.x-1,e.y+9)) then
   e.vx=0.6 e.x=e.x\8*8+8
  end
  if p.iframe==0 and overlap(p.x,p.y,7,8,e.x,e.y,8,8) then
   p.hp-=1 p.iframe=45
   p.vx=(p.x>e.x) and 2 or -2
   p.vy=-1.5
   if p.hp<=0 then respawn() end
  end
 end
end

function respawn()
 p.x=24 p.y=80 p.hp=4 p.iframe=60
 cam.x=0 cam.y=0 cam.tx=0 cam.ty=0
 state="play"
end

function check_exit()
 if p.x>cam.x+127 then
  cam.tx=cam.x+128 cam.ty=cam.y state="trans"
 elseif p.x+7<cam.x then
  cam.tx=cam.x-128 cam.ty=cam.y state="trans"
 elseif p.y>cam.y+127 then
  cam.tx=cam.x cam.ty=cam.y+128 state="trans"
 elseif p.y+7<cam.y then
  cam.tx=cam.x cam.ty=cam.y-128 state="trans"
 end
end

function do_trans()
 local dx=cam.tx-cam.x
 local dy=cam.ty-cam.y
 cam.x+=mid(-tspd,dx,tspd)
 cam.y+=mid(-tspd,dy,tspd)
 if cam.x==cam.tx and cam.y==cam.ty then
  state="play"
 end
end

function pmove()
 if btnp(2) and p.dash_cd==0 and not p.dashing then
  p.dashing=true p.dash_t=dash_dur
  p.dash_cd=dash_cd_max p.vy=0
  p.iframe=max(p.iframe,dash_dur)
 end
 if p.dash_cd>0 then p.dash_cd-=1 end

 if p.dashing then
  p.vx=p.dir*dash_spd p.vy=0
  p.dash_t-=1
  if p.dash_t<=0 then p.dashing=false end
 else
  local dx=0
  if btn(0) then dx=-1 p.dir=-1 end
  if btn(1) then dx=1  p.dir=1  end
  if dx~=0 then
   p.vx=mid(-spd,p.vx+dx*0.5,spd)
  else
   p.vx*=(p.gnd and gfric or afric)
   if abs(p.vx)<0.05 then p.vx=0 end
  end
 end

 if p.gnd then p.coy=coymax
 elseif p.coy>0 then p.coy-=1 end
 if btnp(4) then p.jbf=jbfmax
 elseif p.jbf>0 then p.jbf-=1 end
 if p.jbf>0 then
  if p.coy>0 then
   p.vy=jforce p.coy=0 p.jbf=0 p.djumped=false
  elseif p.djump and not p.djumped and not p.gnd then
   p.vy=jforce*0.85 p.djumped=true p.jbf=0
  end
 end

 if not btn(4) and p.vy<-1 then p.vy+=0.2 end
 if not p.dashing then p.vy=min(p.vy+grav,mxfall) end
 if p.iframe>0 then p.iframe-=1 end

 p.x+=p.vx
 if p.vx>0 and (solid(p.x+7,p.y+1) or solid(p.x+7,p.y+6)) then
  p.x=(p.x+7)\8*8-8 p.vx=0 p.dashing=false
 elseif p.vx<0 and (solid(p.x,p.y+1) or solid(p.x,p.y+6)) then
  p.x=p.x\8*8+8 p.vx=0 p.dashing=false
 end
 p.gnd=false
 p.y+=p.vy
 if p.vy>=0 and (solid(p.x+1,p.y+8) or solid(p.x+6,p.y+8)) then
  p.y=(p.y+8)\8*8-8 p.vy=0 p.gnd=true p.djumped=false
 elseif p.vy<0 and (solid(p.x+1,p.y) or solid(p.x+6,p.y)) then
  p.y=p.y\8*8+8 p.vy=0
 end
end

function overlap(ax,ay,aw,ah,bx,by,bw,bh)
 return ax<bx+bw and ax+aw>bx and ay<by+bh and ay+ah>by
end

function solid(x,y)
 if x<0 or y<0 or x>1023 or y>255 then return true end
 local t=mget(x\8,y\8)
 if t==1 then return true end
 if t==2 then return not p.dashing end
 return false
end
"""

rooms = {
    (0,0): [
        "1111111111111111",
        "1000000000000000",
        "1000000000000000",
        "1000000000000000",
        "1000000000000000",
        "1000000000000000",
        "1000001111000000",
        "1000000000000000",
        "1000000000000000",
        "1001110000000000",
        "1000000000000000",
        "1000000000000000",
        "1000000000000000",
        "1111111111111111",
        "1111111111111111",
        "1111111111111111",
    ],
    (1,0): [
        "1111111111111111",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0001111000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000001110000",
        "0000000000000000",
        "0001110000000000",
        "0000000000000000",
        "0000000000000000",
        "1111111111111111",
        "1111111111111111",
        "1111111111111111",
    ],
    (2,0): [
        "1111111111111111",
        "0000000000000001",
        "0000000000000001",
        "0000000000000001",
        "0000000000000001",
        "0000000000000001",
        "0000000000000001",
        "0000011110000001",
        "0000000000000001",
        "0000000000000001",
        "0000000000000001",
        "1100000000011001",
        "0000000000000001",
        "1111111111111111",
        "1111111111111111",
        "1111111111111111",
    ],
    (3,0): [
        "1111111111111111",
        "0000000220000000",
        "0000000220000000",
        "0000111220000000",
        "0000000220000000",
        "0000000220000000",
        "0000000220000000",
        "0000000220000000",
        "0000000220000000",
        "0000000220111000",
        "0000000220000000",
        "0000000220000000",
        "0000000220000000",
        "1111111111111111",
        "1111111111111111",
        "1111111111111111",
    ],
    (4,0): [
        "1111111111111111",
        "0000000000000001",
        "0000000000000001",
        "0000000000000001",
        "0000000000000001",
        "0000111100000001",
        "0000000000000001",
        "0000000001110001",
        "0000000000000001",
        "0000111100000001",
        "0000000000000001",
        "0000000000000001",
        "0000000000000001",
        "1111111111111111",
        "1111111111111111",
        "1111111111111111",
    ],
}

map_grid = [[0]*128 for _ in range(32)]
for (rx, ry), layout in rooms.items():
    for row in range(16):
        for col in range(16):
            map_grid[ry*16 + row][rx*16 + col] = int(layout[row][col])

map_lines = []
for row in range(32):
    line = "".join(f"{map_grid[row][col]:02x}" for col in range(128))
    assert len(line) == 256
    map_lines.append(line)

tile_rows = [
    "55555555","55555555","55050555","55555555",
    "55555505","55555555","55055555","55555555",
]
cracked_rows = [
    "55555555","55050555","55505555","50555555",
    "55505555","55050555","55555555","55505555",
]
player_rows = [
    "00770000","00770000","07777000","00770000",
    "00770000","07007000","07007000","00000000",
]
enemy_rows = [
    "00990000","09999900","09888900","09999900",
    "00990000","09009000","00000000","00000000",
]

gfx_lines = []
for y in range(128):
    if y < 8:
        line = ("00000000" + tile_rows[y] + cracked_rows[y] +
                player_rows[y] + enemy_rows[y] + "0" * 88)
    else:
        line = "0" * 128
    assert len(line) == 128
    gfx_lines.append(line)

gff_lines = ["0" * 128, "0" * 128]

out_path = "C:/Users/ahilt/PycharmProjects/AI_Tools/games/pico8_project/game.p8"
with open(out_path, "w", newline="\n") as f:
    f.write("pico-8 cartridge // http://www.pico-8.com\n")
    f.write("version 16\n")
    f.write("__lua__\n")
    f.write(lua_code)
    f.write("\n__gfx__\n")
    f.write("\n".join(gfx_lines))
    f.write("\n__gff__\n")
    f.write("\n".join(gff_lines))
    f.write("\n__map__\n")
    f.write("\n".join(map_lines))
    f.write("\n__sfx__\n\n__music__\n\n")

print("done")
