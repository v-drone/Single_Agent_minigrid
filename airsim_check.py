import requests


def save_ports_to_file(ports, filepath):
    with open(filepath, 'w') as file:
        for port in ports:
            file.write(f"{port}\n")


def extend_ports_to_file(ports, filepath):
    with open(filepath, 'a') as file:
        for port in ports:
            file.write(f"{port}\n")


def load_ports_from_file(filepath):
    with open(filepath, 'r') as file:
        ports = [int(i) for i in file.read().split()]
    return ports


def manage_servers():
    data = requests.get("http://192.168.0.104:7575/info").json()
    active_ports = data["available"]
    died_ports = data["died"]
    backup_ports = load_ports_from_file('./backup_ports.txt')
    todo_ports = load_ports_from_file('./todo_ports.txt')

    # Ensure there are always at least 12 active ports if possible
    needed_ports = 20 - len(active_ports)
    ports_to_add = backup_ports[:needed_ports]
    todo_ports.extend(ports_to_add)
    backup_ports = backup_ports[needed_ports:]

    save_ports_to_file(backup_ports, './backup_ports.txt')
    save_ports_to_file(todo_ports, './todo_ports.txt')
    extend_ports_to_file(died_ports, './died_ports.txt')


if __name__ == '__main__':
    manage_servers()
