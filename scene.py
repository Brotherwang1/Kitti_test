import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import pygame
import random
from constants import *
import queue
import numpy as np

WINDOW_WIDTH  = 1000
WINDOW_HEIGHT = 1000

class SynchronyModel(object):
    def __init__(self):
        self.world,self.init_setting = self._make_setting()
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []
        self.frame = None
        self.sensors = []
        self._queues = []
        self._span_player()
        self._span_sensor()

    def __enter__(self):
        # set the sensor listener function
        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

    def __exit__(self, *args, **kwargs):
        # cover the world settings
        self.world.apply_settings(self.init_setting)

    def _make_setting(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        # synchrony model and fixed time step
        init_setting = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 steps  per second
        world.apply_settings(settings)
        return world, init_setting

    def _span_player(self):
        my_vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.bmw.*'))
        location = carla.Location(198, 10, 0.5)
        rotation = carla.Rotation(0, 0, 0)
        transform_vehicle = carla.Transform(location, rotation)
        my_vehicle = self.world.spawn_actor(my_vehicle_bp, transform_vehicle)
        self.actor_list.append(my_vehicle)
        self.player = my_vehicle

    def _span_sensor(self):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('fov', '90')
        transform_sensor = carla.Transform(carla.Location(x=0, y=0, z=40), carla.Rotation(-90, -90, 0))
        my_camera = self.world.spawn_actor(camera_bp, transform_sensor)
        self.actor_list.append(my_camera)
        self.sensors.append(my_camera)


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def main():
    pygame.init()
    display = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()
    with SynchronyModel() as sync_mode:
        while True:
            if should_quit():
                break
            clock.tick()
            snapshot, image_rgb = sync_mode.tick(timeout=2.0)
            fps = round(1.0 / snapshot.timestamp.delta_seconds)
            draw_image(display, image_rgb)
            image_rgb.save_to_disk("secne.png")
            display.blit(
                font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                (8, 10))
            display.blit(
                font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                (8, 28))
            pygame.display.flip()

        print('destroying actors.')
        for actor in sync_mode.actor_list:
            actor.destroy()
        pygame.quit()
        print('done.')

if __name__ == '__main__':
    main()