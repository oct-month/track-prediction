from service import get_server


PORT = 3000

if __name__ == '__main__':
    server = get_server(PORT)
    print(f"start server on port {PORT}...")
    server.serve()
    print("done")
