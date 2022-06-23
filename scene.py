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
from dataexport import *

WINDOW_WIDTH  = 1200
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
        my_vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.bmw*'))
        location = carla.Location(146, 65, 0.5)
        rotation = carla.Rotation(0, 0, 0)
        transform_vehicle = carla.Transform(location, rotation)
        my_vehicle = self.world.spawn_actor(my_vehicle_bp, transform_vehicle)
        self.actor_list.append(my_vehicle)
        self.player = my_vehicle

        my_vehicle_bp1 = random.choice(self.blueprint_library.filter('vehicle.bmw*'))
        location = carla.Location(184, 66, 0.5)
        rotation = carla.Rotation(0, 0, 0)
        transform_vehicle = carla.Transform(location, rotation)
        my_vehicle1 = self.world.spawn_actor(my_vehicle_bp1, transform_vehicle)
        self.actor_list.append(my_vehicle1)

        my_vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.tesla.m*'))
        location = carla.Location(167.5, 107, 0.5)
        rotation = carla.Rotation(0, -60, 0)
        transform_vehicle = carla.Transform(location, rotation)
        my_vehicle2 = self.world.spawn_actor(my_vehicle_bp, transform_vehicle)
        self.actor_list.append(my_vehicle2)

        my_vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.tesla.m*'))
        location = carla.Location(167, 71, 0.5)
        rotation = carla.Rotation(0, 88, 0)
        transform_vehicle = carla.Transform(location, rotation)
        my_vehicle4 = self.world.spawn_actor(my_vehicle_bp, transform_vehicle)
        self.actor_list.append(my_vehicle4)

    def _span_sensor(self):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('fov', '95')
        #transform_sensor = carla.Transform(carla.Location(x=-50, y=20, z=30), carla.Rotation(-90, 0, 0))
        transform_sensor = carla.Transform(carla.Location(x=170, y=60, z=30), carla.Rotation(-70, 90, 0))
        my_camera = self.world.spawn_actor(camera_bp, transform_sensor)


        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(lidar_range))
        lidar_bp.set_attribute('rotation_frequency', str(lidar_rotation_frequency))
        lidar_bp.set_attribute('upper_fov', str(lidar_upper_fov))
        lidar_bp.set_attribute('lower_fov', str(lidar_lower_fov))
        lidar_bp.set_attribute('points_per_second', str(points_per_second))
        lidar_bp.set_attribute('channels', str(channels))

   
        #transform_vehicle = carla.Transform(carla.Location(146, 65, 3), carla.Rotation(0, 0, 0))
        #transform_sensor1 = carla.Transform(carla.Location(x=186, y=69.5, z=2.5), carla.Rotation(0, 0, 0))
        #transform_sensor2 = carla.Transform(carla.Location(x=157.5, y=107, z=2.5), carla.Rotation(0, 0, 0))
        transform_sensor3 = carla.Transform(carla.Location(x=155.5, y=53, z=3), carla.Rotation(0, 0, 0))
        my_lidar = self.world.spawn_actor(lidar_bp, transform_sensor3)

        self.actor_list.append(my_camera)
        self.sensors.append(my_camera)

        self.actor_list.append(my_lidar)
        self.sensors.append(my_lidar)

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
            snapshot, image_rgb, lidar_data = sync_mode.tick(timeout=2.0)
            fps = round(1.0 / snapshot.timestamp.delta_seconds)
            draw_image(display, image_rgb)
            image_rgb.save_to_disk("secne.png")

            points_vehicle = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
            points_vehicle = np.reshape(points_vehicle, (int(points_vehicle.shape[0] / 4), 4))
            save_lidar_data("1.bin", points_vehicle)

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