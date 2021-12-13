"""
Python wrapper for body tracking SDK

Source: https://github.com/hexops/Azure-Kinect-Python

"""


"""
This is a python port of the simple-sample project from.
    https://github.com/microsoft/Azure-Kinect-Samples/blob/bf2f8cf95d969dcc7842c4c450052fe5a943c756/body-tracking-samples/simple_sample/main.c
"""

import traceback
import sys
import ctypes
import os
import time

# Add .. to the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import k4a


class BodyTracker:

    def __init__(self):

        # create and open device:

        self.stop = False

        device_config = k4a.K4A_DEVICE_CONFIG_INIT_DISABLE_ALL
        device_config.depth_mode = k4a.K4A_DEPTH_MODE_NFOV_UNBINNED

        self.device = k4a.k4a_device_t()
        self.VERIFY(k4a.k4a_device_open(0, ctypes.byref(self.device)), "Open K4A Device failed!")
        self.VERIFY(k4a.k4a_device_start_cameras(self.device, ctypes.byref(device_config)), "Start K4A cameras failed!")

        sensor_calibration = k4a.k4a_calibration_t()
        self.VERIFY(k4a.k4a_device_get_calibration(self.device, device_config.depth_mode, k4a.K4A_COLOR_RESOLUTION_OFF,
                                                   ctypes.byref(sensor_calibration)),
                    "Get depth camera calibration failed!")

        self.tracker = k4a.k4abt_tracker_t()
        tracker_config = k4a.K4ABT_TRACKER_CONFIG_DEFAULT
        self.VERIFY(
            k4a.k4abt_tracker_create(ctypes.byref(sensor_calibration), tracker_config, ctypes.byref(self.tracker)),
            "Body tracker initialization failed!")

    def VERIFY(self, result, error):
        if result != k4a.K4A_RESULT_SUCCEEDED:
            print(error)
            traceback.print_stack()
            self.close_device()
            sys.exit(1)

    def close(self):
        """
        close device and abort tracking
        :return:
        """
        self.stop = True

    def frame_parsing(self):
        """
        create a frame with all body joints and append to queue to be used in other threads
        :return:
        """
        pass


    def close_device(self):
        print("Finished body tracking processing!")

        k4a.k4abt_tracker_shutdown(self.tracker)
        k4a.k4abt_tracker_destroy(self.tracker)
        k4a.k4a_device_stop_cameras(self.device)
        k4a.k4a_device_close(self.device)

    def print_body_information(self, body):
        print("Body ID: {}".format(body.id))
        for i in range(k4a.K4ABT_JOINT_COUNT):
            position = body.skeleton.joints[i].position
            orientation = body.skeleton.joints[i].orientation
            confidence_level = body.skeleton.joints[i].confidence_level

            # print spine chest joint - joint 2
            if i == 2:
                print(
                    "Joint[{}]: Position[mm] ( {}, {}, {} ); Orientation ( {}, {}, {}, {}); Confidence Level ({})".format(
                        i, position.v[0], position.v[1], position.v[2], orientation.v[0], orientation.v[1],
                        orientation.v[2],
                        orientation.v[3], confidence_level
                    ))

    def print_body_index_map_middle_line(self, body_index_map):
        print("print_body_index_map_middle_line not implemented")
        """
        uint8_t* body_index_map_buffer = k4a_image_get_buffer(body_index_map);
        // Given body_index_map pixel type should be uint8, the stride_byte should be the same as width
        // TODO: Since there is no API to query the byte-per-pixel information, we have to compare the width and stride to
        // know the information. We should replace this assert with proper byte-per-pixel query once the API is provided by
        // K4A SDK.
        assert(k4a_image_get_stride_bytes(body_index_map) == k4a_image_get_width_pixels(body_index_map));
        int middle_line_num = k4a_image_get_height_pixels(body_index_map) / 2;
        body_index_map_buffer = body_index_map_buffer + middle_line_num * k4a_image_get_width_pixels(body_index_map);
        printf("BodyIndexMap at Line %d:\n", middle_line_num);
        for (int i = 0; i < k4a_image_get_width_pixels(body_index_map); i++)
        {
            printf("%u, ", *body_index_map_buffer);
            body_index_map_buffer++;
        }
        printf("\n");
        """

    def main(self):

        frame_count = 0
        while not self.stop and frame_count<10:
            sensor_capture = k4a.k4a_capture_t()
            get_capture_result = k4a.k4a_device_get_capture(self.device, ctypes.byref(sensor_capture),
                                                            k4a.K4A_WAIT_INFINITE)

            if get_capture_result == k4a.K4A_WAIT_RESULT_SUCCEEDED:
                frame_count += 1

                print("Start processing frame {}".format(frame_count))

                queue_capture_result = k4a.k4abt_tracker_enqueue_capture(self.tracker, sensor_capture,
                                                                         k4a.K4A_WAIT_INFINITE)

                k4a.k4a_capture_release(sensor_capture)

                if queue_capture_result == k4a.K4A_WAIT_RESULT_TIMEOUT:
                    # It should never hit timeout when K4A_WAIT_INFINITE is set.
                    print("Error! Add capture to tracker process queue timeout!")
                    break
                elif queue_capture_result == k4a.K4A_WAIT_RESULT_FAILED:
                    print("Error! Add capture to tracker process queue failed!")
                    break

                body_frame = k4a.k4abt_frame_t()
                pop_frame_result = k4a.k4abt_tracker_pop_result(self.tracker, ctypes.byref(body_frame),
                                                                k4a.K4A_WAIT_INFINITE)
                if pop_frame_result == k4a.K4A_WAIT_RESULT_SUCCEEDED:
                    num_bodies = k4a.k4abt_frame_get_num_bodies(body_frame)
                    print("{} bodies are detected!".format(num_bodies))

                    for i in range(num_bodies):
                        body = k4a.k4abt_body_t()
                        self.VERIFY(k4a.k4abt_frame_get_body_skeleton(body_frame, i, ctypes.byref(body.skeleton)),
                                    "Get body from body frame failed!")
                        body.id = k4a.k4abt_frame_get_body_id(body_frame, i)

                        self.print_body_information(body)

                    body_index_map = k4a.k4abt_frame_get_body_index_map(body_frame)
                    if body_index_map:
                        self.print_body_index_map_middle_line(body_index_map)
                        k4a.k4a_image_release(body_index_map)
                    else:
                        print("Error: Fail to generate bodyindex map!")

                    k4a.k4abt_frame_release(body_frame)
                elif pop_frame_result == k4a.K4A_WAIT_RESULT_TIMEOUT:
                    # It should never hit timeout when K4A_WAIT_INFINITE is set.
                    print("Error! Pop body frame result timeout!")
                    break
                else:
                    print("Pop body frame result failed!")
                    break
            elif get_capture_result == k4a.K4A_WAIT_RESULT_TIMEOUT:
                # It should never hit timeout when K4A_WAIT_INFINITE is set.
                print("Error! Get depth frame time out!")
                break
            else:
                print("Get depth capture returned error: {}".format(get_capture_result))

        # some tests
        # print("-----------------TEST-----------------------")
        # print(body.id)
        # print("-----------------END-----------------------")

        self.close_device()


if __name__ == '__main__':
    device = BodyTracker()
    device.main()
    device.close()
