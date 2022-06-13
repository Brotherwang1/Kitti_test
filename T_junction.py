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
import logging
import math
import pygame
import random
import queue
import numpy as np
from bounding_box import create_kitti_datapoint
from constants import *
import image_converter
from dataexport import *

""" OUTPUT FOLDER GENERATION """

OUTPUT_FOLDER_vehicle = os.path.join("T_junction_pedestrians/vehicle", "training")
OUTPUT_FOLDER_sensor1 = os.path.join("T_junction_pedestrians/sensor1", "training")
OUTPUT_FOLDER_sensor2 = os.path.join("T_junction_pedestrians/sensor2", "training")
folders = ['calib', 'image_2', 'label_2', 'velodyne']

OUTPUT_FOLDER_top_camera = os.path.join("T_junction_pedestrians", "top_image")


def maybe_create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


for folder in folders:
    maybe_create_dir(os.path.join(OUTPUT_FOLDER_vehicle, folder))
    maybe_create_dir(os.path.join(OUTPUT_FOLDER_sensor1, folder))
    maybe_create_dir(os.path.join(OUTPUT_FOLDER_sensor2, folder))

maybe_create_dir(OUTPUT_FOLDER_top_camera)

""" DATA SAVE PATHS """
#GROUNDPLANE_PATH_vehicle = os.path.join(OUTPUT_FOLDER_vehicle, 'planes/{0:06}.txt')
LIDAR_PATH_vehicle = os.path.join(OUTPUT_FOLDER_vehicle, 'velodyne/{0:06}.bin')
LABEL_PATH_vehicle = os.path.join(OUTPUT_FOLDER_vehicle, 'label_2/{0:06}.txt')
IMAGE_PATH_vehicle = os.path.join(OUTPUT_FOLDER_vehicle, 'image_2/{0:06}.png')
CALIBRATION_PATH_vehicle = os.path.join(OUTPUT_FOLDER_vehicle, 'calib/{0:06}.txt')

#GROUNDPLANE_PATH_sensor1 = os.path.join(OUTPUT_FOLDER_sensor1, 'planes/{0:06}.txt')
LIDAR_PATH_sensor1 = os.path.join(OUTPUT_FOLDER_sensor1, 'velodyne/{0:06}.bin')
LABEL_PATH_sensor1 = os.path.join(OUTPUT_FOLDER_sensor1, 'label_2/{0:06}.txt')
IMAGE_PATH_sensor1 = os.path.join(OUTPUT_FOLDER_sensor1, 'image_2/{0:06}.png')
CALIBRATION_PATH_sensor1 = os.path.join(OUTPUT_FOLDER_sensor1, 'calib/{0:06}.txt')


#GROUNDPLANE_PATH_sensor2 = os.path.join(OUTPUT_FOLDER_sensor2, 'planes/{0:06}.txt')
LIDAR_PATH_sensor2 = os.path.join(OUTPUT_FOLDER_sensor2, 'velodyne/{0:06}.bin')
LABEL_PATH_sensor2 = os.path.join(OUTPUT_FOLDER_sensor2, 'label_2/{0:06}.txt')
IMAGE_PATH_sensor2 = os.path.join(OUTPUT_FOLDER_sensor2, 'image_2/{0:06}.png')
CALIBRATION_PATH_sensor2 = os.path.join(OUTPUT_FOLDER_sensor2, 'calib/{0:06}.txt')



IMAGE_PATH_top_image = os.path.join(OUTPUT_FOLDER_top_camera, 'image_2/{0:06}.png')




class SynchronyModel(object):
    def __init__(self):
        self.world, self.init_setting, self.client, self.traffic_manager = self._make_setting()
        self.blueprint_library = self.world.get_blueprint_library()
        self.non_player = []
        self.actor_list = []
        self.frame = None
        self.captured_frame_no = self.current_captured_frame_num()

        self.player_vehicle = None
        self.player_sensor1 = None
        self.player_sensor2 = None
        self.sensors = []
        self._queues = []

        self.main_image_vehicle = None
        self.depth_image_vehicle = None
        self.point_cloud_vehicle = None
        self.extrinsic_vehicle = None

        self.main_image_sensor1 = None
        self.depth_image_sensor1 = None
        self.point_cloud_sensor1 = None
        self.extrinsic_sensor1 = None

        self.main_image_sensor2 = None
        self.depth_image_sensor2 = None
        self.point_cloud_sensor2 = None
        self.extrinsic_sensor2 = None

        self.main_image_top_camera = None

        self.intrinsic_vehicle, self.my_camera_vehicle, self.intrinsic_sensor1, self.my_camera_sensor1, \
        self.intrinsic_sensor2, self.my_camera_sensor2, self.intrinsic_top_camera, self.top_camera = self._span_player()
        self._span_non_player()

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

    def current_captured_frame_num(self):
        # Figures out which frame number we currently are on
        # This is run once, when we start the simulator in case we already have a dataset.
        # The user can then choose to overwrite or append to the dataset.
        label_path = os.path.join(OUTPUT_FOLDER_vehicle, 'label_2/')
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith('.txt')])
        print(num_existing_data_files)
        if num_existing_data_files == 0:
            return 0
        answer = input(
            "There already exists a dataset in {}. Would you like to (O)verwrite or (A)ppend the dataset? (O/A)".format(
                OUTPUT_FOLDER_vehicle))
        if answer.upper() == "O":
            logging.info(
                "Resetting frame number to 0 and overwriting existing")
            # Overwrite the data
            return 0
        logging.info("Continuing recording data on frame number {}".format(
            num_existing_data_files))
        return num_existing_data_files

    def tick(self, timeout):
        # Drive the simulator to the next frame and get the data of the current frame
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
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        traffic_manager.set_synchronous_mode(True)
        # synchrony model and fixed time step
        init_setting = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 steps  per second
        world.apply_settings(settings)
        return world, init_setting, client, traffic_manager

    def _span_player(self):
        """create our target vehicle"""
        my_vehicle_bp = random.choice(self.blueprint_library.filter("vehicle"))
        location_vehicle = carla.Location(174, 60, 2)
        rotation_vehicle = carla.Rotation(0, 0, 0)
        transform_vehicle = carla.Transform(location_vehicle, rotation_vehicle)
        #my_vehicle = self.world.spawn_actor(my_vehicle_bp, transform_vehicle)
        #k1, my_camera1 = self._span_sensor(my_vehicle)
        k1, my_camera1 = self._span_infrastructures_sensor(transform_vehicle)


        """create our infrastructures"""
        location_sensor1 = carla.Location(235, 20, 2)
        rotation_sensor1 = carla.Rotation(0, 90, 0)
        transform_sensor1 = carla.Transform(location_sensor1, rotation_sensor1)
        k2, my_camera2 = self._span_infrastructures_sensor(transform_sensor1)

        location_sensor2 = carla.Location(235, 100, 2)
        rotation_sensor2 = carla.Rotation(0, -90, 0)
        transform_sensor2 = carla.Transform(location_sensor2, rotation_sensor2)
        k3, my_camera3 = self._span_infrastructures_sensor(transform_sensor2)



        location_top_camera = carla.Location(210, 60, 40)
        rotation_top_camera  = carla.Rotation(-90, -90, 0)
        transform_top_camera = carla.Transform(location_top_camera , rotation_top_camera )
        k_top, top_camera = self._span_top_camera(transform_top_camera)


        #self.actor_list.append(my_vehicle)

        #self.player_vehicle = my_vehicle
        self.player_vehicle = my_camera1
        

        self.player_sensor1 = my_camera2
        self.player_sensor2 = my_camera3



        return k1, my_camera1, k2, my_camera2, k3, my_camera3, k_top, top_camera

    def _span_sensor(self, player):
        """create camera, depth camera and lidar and attach to the target vehicle"""
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_d_bp = self.blueprint_library.find('sensor.camera.depth')
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')

        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('fov', '90')

        camera_d_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_d_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_d_bp.set_attribute('fov', '90')

        lidar_bp.set_attribute('range', str(lidar_range))
        lidar_bp.set_attribute('rotation_frequency', str(lidar_rotation_frequency))
        lidar_bp.set_attribute('upper_fov', str(lidar_upper_fov))
        lidar_bp.set_attribute('lower_fov', str(lidar_lower_fov))
        lidar_bp.set_attribute('points_per_second', str(points_per_second))
        lidar_bp.set_attribute('channels', str(channels))

        transform_sensor = carla.Transform(carla.Location(x=0, y=0, z=CAMERA_HEIGHT_POS))

        my_camera = self.world.spawn_actor(camera_bp, transform_sensor, attach_to=player)
        my_camera_d = self.world.spawn_actor(camera_d_bp, transform_sensor, attach_to=player)
        my_lidar = self.world.spawn_actor(lidar_bp, transform_sensor, attach_to=player)

        self.actor_list.append(my_camera)
        self.actor_list.append(my_camera_d)
        self.actor_list.append(my_lidar)
        self.sensors.append(my_camera)
        self.sensors.append(my_camera_d)
        self.sensors.append(my_lidar)

        # camera intrinsic
        k = np.identity(3)
        k[0, 2] = WINDOW_WIDTH_HALF
        k[1, 2] = WINDOW_HEIGHT_HALF
        f = WINDOW_WIDTH / (2.0 * math.tan(90.0 * math.pi / 360.0))
        k[0, 0] = k[1, 1] = f
        return k, my_camera
    
    def _span_top_camera(self, transform_sensor):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        
        camera_bp.set_attribute('image_size_x', str(2000))
        camera_bp.set_attribute('image_size_y', str(2000))
        camera_bp.set_attribute('fov', '90')

        my_top_camera = self.world.spawn_actor(camera_bp, transform_sensor)
        self.actor_list.append(my_top_camera)
        self.sensors.append(my_top_camera)

        # camera intrinsic
        k = np.identity(3)
        k[0, 2] = WINDOW_WIDTH_HALF
        k[1, 2] = WINDOW_HEIGHT_HALF
        f = WINDOW_WIDTH / (2.0 * math.tan(90.0 * math.pi / 360.0))
        k[0, 0] = k[1, 1] = f
        return k, my_top_camera
    
    def _span_infrastructures_sensor(self, transform_sensor):
        """create camera, depth camera and lidar and attach to the target vehicle"""
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_d_bp = self.blueprint_library.find('sensor.camera.depth')
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')

        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('fov', '90')

        camera_d_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_d_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_d_bp.set_attribute('fov', '90')

        lidar_bp.set_attribute('range', str(lidar_range))
        lidar_bp.set_attribute('rotation_frequency', str(lidar_rotation_frequency))
        lidar_bp.set_attribute('upper_fov', str(lidar_upper_fov))
        lidar_bp.set_attribute('lower_fov', str(lidar_lower_fov))
        lidar_bp.set_attribute('points_per_second', str(points_per_second))
        lidar_bp.set_attribute('channels', str(channels))

        my_camera = self.world.spawn_actor(camera_bp, transform_sensor)
        my_camera_d = self.world.spawn_actor(camera_d_bp, transform_sensor)
        my_lidar = self.world.spawn_actor(lidar_bp, transform_sensor)

        self.actor_list.append(my_camera)
        self.actor_list.append(my_camera_d)
        self.actor_list.append(my_lidar)
        self.sensors.append(my_camera)
        self.sensors.append(my_camera_d)
        self.sensors.append(my_lidar)

        # camera intrinsic
        k = np.identity(3)
        k[0, 2] = WINDOW_WIDTH_HALF
        k[1, 2] = WINDOW_HEIGHT_HALF
        f = WINDOW_WIDTH / (2.0 * math.tan(90.0 * math.pi / 360.0))
        k[0, 0] = k[1, 1] = f
        return k, my_camera

    def _span_non_player(self):
        """create autonomous vehicles and people"""
        blueprints = self.world.get_blueprint_library().filter(FILTERV)
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        '''
        spawn_points = []
        for i in range(NUM_OF_WALKERS):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            while(loc.x<150 or loc.y>100 or loc.y<0):
                loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        '''
        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        if NUM_OF_VEHICLES < number_of_spawn_points:
            random.shuffle(spawn_points)
            number_of_vehicles = NUM_OF_VEHICLES
        elif NUM_OF_VEHICLES > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, NUM_OF_VEHICLES, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points
      
        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor

        batch = []
        n = 0
        for _, transform in enumerate(spawn_points):
            loc = transform.location
            if (loc.x<150 or loc.y>130 or loc.y<-130):
                continue
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot
            batch.append(SpawnActor(blueprint, transform))
            n += 1

        vehicles_id = []
        for response in self.client.apply_batch_sync(batch):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_id.append(response.actor_id)
        vehicle_actors = self.world.get_actors(vehicles_id)
        self.non_player.extend(vehicle_actors)
        self.actor_list.extend(vehicle_actors)

        for i in vehicle_actors:
            i.set_autopilot(True, self.traffic_manager.get_port())

        blueprintsWalkers = self.world.get_blueprint_library().filter(FILTERW)
        percentagePedestriansRunning = 0.0  # how many pedestrians will run
        percentagePedestriansCrossing = 0.80  # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(NUM_OF_WALKERS):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            while(loc.x<170 or loc.y>130 or loc.y<10):
                loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        walkers_list = []
        all_id = []
        walkers_id = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = self.world.get_actors(all_id)

        for i in range(len(walkers_list)):
            walkers_id.append(walkers_list[i]["id"])
        walker_actors = self.world.get_actors(walkers_id)
        self.non_player.extend(walker_actors)
        self.actor_list.extend(all_actors)
        self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        print('spawned %d walkers and %d vehicles, press Ctrl+C to exit.' % (len(walkers_id), len(vehicles_id)))

    def _save_training_files(self, datapoints, point_cloud, image, GROUNDPLANE_PATH, LIDAR_PATH, LABEL_PATH, IMAGE_PATH, CALIBRATION_PATH, OUTPUT_FOLDER, player, intrinsic, extrinsic):
        """ Save data in Kitti dataset format """
        logging.info("Attempting to save at frame no {}, frame no: {}".format(self.frame, self.captured_frame_no))
        #groundplane_fname = GROUNDPLANE_PATH.format(self.captured_frame_no)
        lidar_fname = LIDAR_PATH.format(self.captured_frame_no)
        kitti_fname = LABEL_PATH.format(self.captured_frame_no)
        img_fname = IMAGE_PATH.format(self.captured_frame_no)
        calib_filename = CALIBRATION_PATH.format(self.captured_frame_no)

        #save_groundplanes(
            #groundplane_fname, player, LIDAR_HEIGHT_POS)
        #save_ref_files(OUTPUT_FOLDER, self.captured_frame_no)
        save_image_data(
            img_fname, image)
        save_kitti_data(kitti_fname, datapoints)

        save_calibration_matrices(
            calib_filename, intrinsic, extrinsic)

        save_lidar_data(lidar_fname, point_cloud)

    def generate_datapoints(self, image_vehicle, image_sensor1, image_sensor2):
        """ Returns a list of datapoints (labels and such) that are generated this frame together with the main image
        image """

        datapoints_vehicle = []
        image_vehicle = image_vehicle.copy()

        datapoints_sensor1 = []
        image_sensor1 = image_sensor1.copy()

        datapoints_sensor2 = []
        image_sensor2 = image_sensor2.copy()



        
        # Remove this
        rotRP = np.identity(3)
        # Stores all datapoints for the current frames
        for agent in self.non_player:
            if GEN_DATA:
                image_vehicle, kitti_datapoint_vehicle = create_kitti_datapoint(
                    agent, self.intrinsic_vehicle, self.extrinsic_vehicle, image_vehicle, self.depth_image_vehicle, self.player_vehicle, rotRP, draw_3D_bbox=False)

                image_sensor1, kitti_datapoint_sensor1 = create_kitti_datapoint(
                    agent, self.intrinsic_sensor1, self.extrinsic_sensor1, image_sensor1, self.depth_image_sensor1, self.player_sensor1, rotRP, draw_3D_bbox=False)

                image_sensor2, kitti_datapoint_sensor2 = create_kitti_datapoint(
                    agent, self.intrinsic_sensor2, self.extrinsic_sensor2, image_sensor2, self.depth_image_sensor2, self.player_sensor2, rotRP, draw_3D_bbox=False)

        
                if kitti_datapoint_vehicle == []:
                    kitti_datapoint_vehicle = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
             
                datapoints_vehicle.append(kitti_datapoint_vehicle)

                if kitti_datapoint_sensor1 == []:
                    kitti_datapoint_sensor1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                
                datapoints_sensor1.append(kitti_datapoint_sensor1)
                
                if kitti_datapoint_sensor2 == []:
                    kitti_datapoint_sensor2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
           
                datapoints_sensor2.append(kitti_datapoint_sensor2)
                

        return image_vehicle, datapoints_vehicle, image_sensor1, datapoints_sensor1, image_sensor2, datapoints_sensor2

def draw_image(surface, image, blend=False):
    # array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    # array = np.reshape(array, (image.height, image.width, 4))
    # array = array[:, :, :3]
    # array = array[:, :, ::-1]
    array = image[:, :, ::-1]
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
        try:
            step = 1
            while True:
                if should_quit():
                    break
                clock.tick()
                snapshot, sync_mode.main_image_vehicle, sync_mode.depth_image_vehicle, sync_mode.point_cloud_vehicle, sync_mode.main_image_sensor1, sync_mode.depth_image_sensor1, sync_mode.point_cloud_sensor1, \
                sync_mode.main_image_sensor2, sync_mode.depth_image_sensor2, sync_mode.point_cloud_sensor2, sync_mode.main_image_top_camera = sync_mode.tick(timeout=2.0)
                sync_mode.extrinsic_vehicle = np.mat(sync_mode.my_camera_vehicle.get_transform().get_matrix())
                sync_mode.extrinsic_sensor1 = np.mat(sync_mode.my_camera_sensor1.get_transform().get_matrix())
                sync_mode.extrinsic_sensor2 = np.mat(sync_mode.my_camera_sensor2.get_transform().get_matrix())
          

                image_vehicle = image_converter.to_rgb_array(sync_mode.main_image_vehicle)
                image_sensor1 = image_converter.to_rgb_array(sync_mode.main_image_sensor1)
                image_sensor2 = image_converter.to_rgb_array(sync_mode.main_image_sensor2)
      

                
                image_vehicle, datapoints_vehicle, image_sensor1, datapoints_sensor1, image_sensor2, datapoints_sensor2 = sync_mode.generate_datapoints(image_vehicle, image_sensor1, image_sensor2)
                
             
               
                if (datapoints_vehicle or datapoints_sensor1 or datapoints_sensor2) and step % 71 == 0:
                    points_vehicle = np.copy(np.frombuffer(sync_mode.point_cloud_vehicle.raw_data, dtype=np.dtype('f4')))
                    points_vehicle = np.reshape(points_vehicle, (int(points_vehicle.shape[0] / 4), 4))
                    # Isolate the 3D data
                    #points_vehicle = data_vehicle
                    # transform to car space
                    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
                    # points = np.dot(sync_mode.player.get_transform().get_matrix(), points.T).T
                    # points = points[:, :-1]
                    # points[:, 2] -= LIDAR_HEIGHT_POS

                    points_sensor1 = np.copy(np.frombuffer(sync_mode.point_cloud_sensor1.raw_data, dtype=np.dtype('f4')))
                    points_sensor1 = np.reshape(points_sensor1, (int(points_sensor1.shape[0] / 4), 4))
              
                    points_sensor2 = np.copy(np.frombuffer(sync_mode.point_cloud_sensor2.raw_data, dtype=np.dtype('f4')))
                    points_sensor2 = np.reshape(points_sensor2, (int(points_sensor2.shape[0] / 4), 4))
          

                    sync_mode._save_training_files(datapoints_vehicle, points_vehicle, sync_mode.main_image_vehicle, GROUNDPLANE_PATH_vehicle, LIDAR_PATH_vehicle, LABEL_PATH_vehicle, IMAGE_PATH_vehicle, CALIBRATION_PATH_vehicle, OUTPUT_FOLDER_vehicle, sync_mode.player_vehicle, sync_mode.intrinsic_vehicle, sync_mode.extrinsic_vehicle)
                    sync_mode._save_training_files(datapoints_sensor1, points_sensor1, sync_mode.main_image_sensor1, GROUNDPLANE_PATH_sensor1, LIDAR_PATH_sensor1, LABEL_PATH_sensor1, IMAGE_PATH_sensor1, CALIBRATION_PATH_sensor1, OUTPUT_FOLDER_sensor1, sync_mode.player_sensor1, sync_mode.intrinsic_sensor1, sync_mode.extrinsic_sensor1)
                    sync_mode._save_training_files(datapoints_sensor2, points_sensor2, sync_mode.main_image_sensor2, GROUNDPLANE_PATH_sensor2, LIDAR_PATH_sensor2, LABEL_PATH_sensor2, IMAGE_PATH_sensor2, CALIBRATION_PATH_sensor2, OUTPUT_FOLDER_sensor2, sync_mode.player_sensor2, sync_mode.intrinsic_sensor2, sync_mode.extrinsic_sensor2)
                   
                    img_fname = IMAGE_PATH_top_image.format(sync_mode.captured_frame_no)
                    save_image_data(img_fname, sync_mode.main_image_top_camera)
                    print(sync_mode.captured_frame_no)
                    sync_mode.captured_frame_no += 1
                    step = 1
                
                
                #image_top = image_converter.to_rgb_array(sync_mode.main_image_top_camera)

                step = step+1
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                draw_image(display, image_vehicle)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()
        finally:
            print('destroying actors.')
            for actor in sync_mode.actor_list:
                actor.destroy()
            pygame.quit()
            print('done.')


if __name__ == '__main__':
    main()
