import overpy

location = "-1.4927,55.0032,-1.4685,55.0124"
# location = "50.745,7.17,50.75,7.18"
api = overpy.Overpass()

# fetch all ways and nodes
result = api.query("""node(%s);out;""" % location)

for way in result.ways:
    print("Name: %s" % way.tags.get("name", "n/a"))
    print("  Nodes:")
    for node in way.nodes:
        print("    Lat: %f, Lon: %f" % (node.lat, node.lon))
