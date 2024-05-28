import requests
import logging
import json


def check_server(port):
    try:
        response = requests.get(f"http://127.0.0.1:{port}/ping", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logging.error(f"Ping failed: {str(e)}")
        return False


def manage_servers(filename='./server_ports.json'):
    # 读取或初始化端口信息
    active_ports = requests.get(f"http://127.0.0.1:7575/info").json()["available"]
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            backup_ports = data['backup_ports']
            todo_ports = data.get('todo_ports', [])
    except FileNotFoundError:
        # 初始端口配置
        backup_ports = [5008, 5009, 5010, 5011, 5012, 5013, 5014, 5015, 5016, 5017, 5018, 5019]
        todo_ports = []

    # 检查活跃端口并更新列表
    for port in active_ports[:]:
        if not check_server(port):
            try:
                requests.post(f"http://127.0.0.1:{port}/exit")  # 假设这个命令能正常关闭服务
            except Exception as e:
                logging.error(f"Exit failed: {str(e)}")
            active_ports.remove(port)
            if backup_ports:
                new_port = backup_ports.pop(0)
                todo_ports.append(new_port)
                active_ports.append(new_port)

    # 保存更新后的端口信息
    data = {
        'active_ports': active_ports,
        'backup_ports': backup_ports,
        'todo_ports': todo_ports
    }
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == '__main__':
    manage_servers()
