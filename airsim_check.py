import requests


def save_ports_to_file(ports, filename):
    with open(filename, 'w') as file:
        for port in ports:
            file.write(f"{port}\n")


def load_ports_from_file(filename):
    try:
        with open(filename, 'r') as file:
            return [int(line.strip()) for line in file if line.strip()]
    except FileNotFoundError:
        return []


def manage_servers():
    data = requests.get("http://192.168.0.104:7575/info").json()
    active_ports = data["available"]
    died_ports = data["died"]
    backup_ports = load_ports_from_file('./backup_ports.txt')
    todo_ports = load_ports_from_file('./todo_ports.txt')

    if len(active_ports) < 12:
        if backup_ports:
            new_port = backup_ports.pop(0)
            todo_ports.append(new_port)
            requests.post('http://192.168.0.104:7575/add', json={'port': new_port})

    save_ports_to_file(backup_ports, './backup_ports.txt')
    save_ports_to_file(todo_ports, './todo_ports.txt')
    save_ports_to_file(died_ports, './died_ports.txt')


if __name__ == '__main__':
    manage_servers()
