import socket, json
import requests
import time

class Test:
    def __init__(self):
        self.HOST, self.PORT = "192.168.10.10", 31001
        self.base_url = f"http://{self.HOST}:{self.PORT}"
        self.move_api = "/api/move?marker={target}"
        self.status_api = "/api/robot_status"

    def read_info_loop(self, poll_interval=1.0, timeout=60):
        try:
            command = self.status_api
            # Create a TCP/IP socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # Connect to the server
                sock.connect((self.HOST, self.PORT))
                start_time = time.time()
                while time.time() - start_time < timeout:
                    
                    
                    # Send the command
                    sock.sendall(command.encode('utf-8'))
                    
                    # Receive the response
                    response = sock.recv(4096).decode('utf-8')
                    
                    # Parse the JSON response
                    response_json = json.loads(response)
                    move_status = response_json.get('results', {}).get('current_pose', None)
                    print("RESP:", move_status)
                    time.sleep(poll_interval)
        except Exception as e:
            print(f"指令发送失败: {e}")
            return False
    
if __name__ == "__main__":
    tester = Test()
    tester.read_info_loop()
