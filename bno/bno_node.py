import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import serial

class IMUSerialDriver(Node):
    def __init__(self):
        super().__init__('imu_serial_driver')
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('frame_id', 'imu_link')
        
        port = self.get_parameter('port').value
        baudrate = self.get_parameter('baudrate').value
        self.frame_id = self.get_parameter('frame_id').value
        
        self.serial_port = serial.Serial(port, baudrate, timeout=0.05)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        
        self.timer = self.create_timer(0.01, self.poll_serial_buffer)

    def poll_serial_buffer(self):
        try:
            if self.serial_port.in_waiting > 0:
                try:
                    raw_payload = self.serial_port.readline().decode('utf-8').strip()
                    self.deserialize_payload(raw_payload)
                except UnicodeDecodeError:
                    pass
                except ValueError:
                    pass
        except serial.SerialException:
            pass

    def deserialize_payload(self, payload):
        data_vector = payload.split(',')
        if len(data_vector) == 6:
            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            
            msg.linear_acceleration.x = float(data_vector[0])
            msg.linear_acceleration.y = float(data_vector[1])
            msg.linear_acceleration.z = float(data_vector[2])
            
            msg.angular_velocity.x = float(data_vector[3])
            msg.angular_velocity.y = float(data_vector[4])
            msg.angular_velocity.z = float(data_vector[5])
            
            msg.orientation_covariance[0] = -1.0
            
            self.imu_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = IMUSerialDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()