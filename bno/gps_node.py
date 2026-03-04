import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
import serial

class GNSSSerialDriver(Node):
    def __init__(self):
        super().__init__('gnss_serial_driver')
        self.declare_parameter('port', '/dev/ttyUSB1')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('frame_id', 'gps_link')

        port = self.get_parameter('port').value
        baudrate = self.get_parameter('baudrate').value
        self.frame_id = self.get_parameter('frame_id').value

        self.gps_pub = self.create_publisher(NavSatFix, '/gps/fix', 10)

        try:
            self.serial_port = serial.Serial(port, baudrate, timeout=0.05)
            self.get_logger().info("GNSS serial interface established.")
        except Exception:
            self.serial_port = None
            self.get_logger().error("GNSS serial interface failure. Verify port allocation.")

        self.timer = self.create_timer(0.05, self.poll_serial_buffer)

    def poll_serial_buffer(self):
        if not self.serial_port:
            return
        try:
            if self.serial_port.in_waiting > 0:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('$GNGGA') or line.startswith('$GPGGA'):
                    self.deserialize_gga_payload(line)
        except Exception:
            pass

    def deserialize_gga_payload(self, payload):
        tokens = payload.split(',')
        
        if len(tokens) < 15:
            return

        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        if tokens[6] == '0':
            msg.status.status = NavSatStatus.STATUS_NO_FIX
            self.gps_pub.publish(msg)
            return

        msg.latitude = self.convert_nmea_to_decimal_degrees(tokens[2], tokens[3])
        msg.longitude = self.convert_nmea_to_decimal_degrees(tokens[4], tokens[5])
        
        try:
            msg.altitude = float(tokens[9])
        except ValueError:
            msg.altitude = 0.0

        msg.status.status = NavSatStatus.STATUS_FIX
        msg.status.service = NavSatStatus.SERVICE_GPS
        
        try:
            hdop = float(tokens[8])
            variance = hdop * hdop
            msg.position_covariance[0] = variance
            msg.position_covariance[4] = variance
            msg.position_covariance[8] = variance * 2.0
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
        except ValueError:
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

        self.gps_pub.publish(msg)

    def convert_nmea_to_decimal_degrees(self, raw_value, cardinal_direction):
        if not raw_value:
            return 0.0
        
        decimal_index = raw_value.find('.')
        if decimal_index == -1:
            return 0.0
            
        degrees_str = raw_value[:decimal_index - 2]
        minutes_str = raw_value[decimal_index - 2:]
        
        degrees = float(degrees_str)
        minutes = float(minutes_str)
        
        decimal_degrees = degrees + (minutes / 60.0)
        
        if cardinal_direction in ['S', 'W']:
            decimal_degrees = -decimal_degrees
            
        return decimal_degrees

def main(args=None):
    rclpy.init(args=args)
    node = GNSSSerialDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()