// Step 1. Create filters

// Center the map on bangka.
var bound = aoi.bounds();
Map.centerObject(bound, 8);

//Construct start and end dates:
var start = ee.Date('2017-01-01');
var finish = ee.Date('2017-12-31');

// Step 5. Load Sentinel 2 dataset
// Function to mask clouds using the Sentinel-2 QA band.
function maskS2clouds(image) {
  var qa = image.select('QA60')
 
  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
             qa.bitwiseAnd(cirrusBitMask).eq(0))

  // Return the masked and scaled data, without the QA bands.
  return image.updateMask(mask).divide(10000)
      .select("B.*")
      .copyProperties(image, ["system:time_start"])
}

// Map the function over one year of data and take the median.
// Load Sentinel-2 TOA reflectance data.
var collection = ee.ImageCollection('COPERNICUS/S2')
    .filterDate(start, finish)
    .filterBounds(bound)
    // Pre-filter to get less cloudy granules.
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskS2clouds);

var composite = collection.median().clip(aoi);

// Center the map on indonesia.
Map.centerObject(bound, 4);

// Display indonesia bounding box.
//Map.addLayer(bound, {}, 'Bounding Box');
//Map.addLayer(composite, covisparam, 'composite');

// step 3. create a land mask

// Load or import the Hansen et al. forest change dataset.
var hansenImage = ee.Image('UMD/hansen/global_forest_change_2015');

// Select the land/water mask.
var datamask = hansenImage.select('datamask');

// Create a binary mask.
var maskland = datamask.eq(2);

// Update the composite mask with the water mask.
var maskedComposite = composite.updateMask(maskland);

// Display the masked composite
//Map.addLayer(maskedComposite, covisparam, 'masked');

// Step 5. Apply Lyzenga algorithm


// Select bands for calculation
var band2 = maskedComposite.select('B2');
var band3 = maskedComposite.select('B3');
var band23 = maskedComposite.select('B2', 'B3');

// Calculate region statistics

// Calculate band variance.
var band2_mean = band2.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: coral_training.geometry(),
  scale: 10,
  maxPixels: 1e8
});

var band2_var = band2.reduceRegion({
  reducer: ee.Reducer.variance(),
  geometry: coral_training.geometry(),
  scale: 10,
  maxPixels: 1e8
});

var band3_mean = band3.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: coral_training.geometry(),
  scale: 10,
  maxPixels: 1e8
});

var band3_var = band3.reduceRegion({
  reducer: ee.Reducer.variance(),
  geometry: coral_training.geometry(),
  scale: 10,
  maxPixels: 1e8
});

// Calculate band covariance
var arrays = band23.toArray();
var covar = arrays.reduceRegion({
  reducer: ee.Reducer.covariance(),
  geometry: coral_training,
  scale: 10,
  maxPixels: 1e8
  });

var band23_covar = ee.Array(covar.get('array')).get([1, 0]);
print('band23_covar',band23_covar);
var b2 = ee.Array(covar.get('array')).get([0, 0]);
var b3 = ee.Array(covar.get('array')).get([1, 1]);

// Calculate a 

var diffvar = ee.Number(b2.getInfo()-b3.getInfo());
print('diffvar', diffvar);

var a23 = diffvar.divide(band23_covar.multiply(2));
print('a23',a23);

// Calculate ki/kj

var a23pow = a23.pow(2);
print('a23pow',a23pow);
var a23pow1 = a23pow.add(1);
print('a23pow1',a23pow1);
var a23pow1root = a23pow1.sqrt();
print('a23pow1root',a23pow1root);
var kikj = a23.add(a23pow1root)
print('kikj',kikj);

// calculate log of bands
var logb2 = maskedComposite.select('B2').log();
print('logb2',logb2);
var logb3 = maskedComposite.select('B3').log();
print('logb3',logb3);
var kikjlogb3 = logb3.multiply(kikj);
print('kikjlogb3',kikjlogb3);

// create new lyzenga band
var lyzenga = ee.Image(ee.Algorithms.If(maskedComposite.select('B4')
  .gt(20), logb2.add(kikjlogb3),'')).rename('lyzenga');

// display lyzenga
//Map.addLayer(lyzenga, {},'Lyzenga'); 

// stack layers
var stacked = maskedComposite.select('B1','B2','B3','B4').addBands(lyzenga);

// Step 7. Supervised classification

var bands = ['B1','B2','B3','B4','lyzenga'];

// merge geometry of training data
var habitat_water = coral_training.merge(sand_training).merge(deep_training);
print(habitat_water, 'habitat_water');

var habitat = coral_training.merge(sand_training);
print(habitat, 'habitat');

// Sample the input imagery to get a FeatureCollection of training data.
var training = stacked.select(bands).sampleRegions({
  collection: habitat_water,
  properties: ['habitat'],
  scale: 10  // should reflect the scale of your imagery
});

// Make a random forest classifier and train it.
var rfclassifier = ee.Classifier.smileRandomForest().train({
  features: training,
  classProperty: 'habitat',
  inputProperties: bands
});

// Make a svm classifier and train it.
var svmclassifier = ee.Classifier.libsvm().train({
  features: training,
  classProperty: 'habitat',
  inputProperties: bands
});

// Classify the input imagery using random forest.
var classifiedrf = stacked.select(bands).classify(rfclassifier);

// Classify the input imagery using svm.
var classifiedsvm = stacked.select(bands).classify(svmclassifier);

// Define a palette for the classification.
var palette = [
  '0800ff', // deep (0)
  'd30000', // coral (1)
  'fff200' //  sand (2)
];

// Display the classifiedrf 
//Map.addLayer(classifiedrf, {min: 0, max: 2, palette: palette}, 'Benthic habitat RF');
//Map.addLayer(classifiedsvm, {min: 0, max: 2, palette: palette}, 'Benthic habitat SVM');


// Get a confusion matrix representing resubstitution accuracy.
var rftrainAccuracy = rfclassifier.confusionMatrix();
print('Rf Resubstitution error matrix: ', rftrainAccuracy);
print('Rf Training overall accuracy: ', rftrainAccuracy.accuracy());

var svmtrainAccuracy = svmclassifier.confusionMatrix();
print('SVM Resubstitution error matrix: ', svmtrainAccuracy);
print('SVM Training overall accuracy: ', svmtrainAccuracy.accuracy());

Export.image.toDrive({
  image: classifiedrf,
  description: 'rote/classifiedrf',
  maxPixels: 1e11,
  scale: 10,
  region: bound
});
Export.image.toDrive({
  image: classifiedsvm,
  description: 'rote/classifiedsvm',
  maxPixels: 1e11,
  scale: 10,
  region: bound
});
