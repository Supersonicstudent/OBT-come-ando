import ee
# Initialize Earth Engine
ee.Initialize()
# Define the date range
start_date = '2020-01-01'
end_date = '2020-12-31'

# Select Landsat 8 imagery
image_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

# Filter the image collection by date
image = image_collection.filterDate(start_date, end_date).first()
# Create a Map object
Map = ee.Map(zoom=6, center=(40, -120))

# Add the Landsat 8 image to the map
Map.addLayer(image, {'bands': ['B5', 'B4', 'B3']}, 'Landsat 8')



