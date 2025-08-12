#/**** Start of imports. If edited, may not auto-convert in the playground. ****/
var table = ee.FeatureCollection("users/YaoLiCD/Potain_1");
#/***** End of imports. If edited, may not auto-convert in the playground. *****/
var roi = table
Map.centerObject(roi, 0);
var cloudMaskL457 = function(image) {
  var qa = image.select('pixel_qa');
  #// If the cloud bit (5) is set and the cloud confidence (7) is high
  #// or the cloud shadow bit is set (3), then it's a bad pixel.
  var cloud = qa.bitwiseAnd(1 << 5)
          .and(qa.bitwiseAnd(1 << 7))
          .or(qa.bitwiseAnd(1 << 3))
  #// Remove edge pixels that don't occur in all bands
  var mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()).updateMask(mask2);
};

var l5 = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
#//var l7 = ee.ImageCollection('LANDSAT/LE07/C01/T1')
#//var L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
var l5Imgs = l5.filterBounds(roi)
               .filterDate("1988-1-01", "1988-12-01")
               .filter(ee.Filter.lte('CLOUD_COVER',15))
               .select('B[1-4]')
               //.map(cloudMaskL457)
print("l5Imgs", l5Imgs);
Map.addLayer(l5Imgs, {min:0, max:2000, bands:["B4","B3","B2"]}, "l5Imgs");
Map.addLayer(roi, {color: "green"}, "roi");



function exportImageCollection(imgCol) {
  var indexList = imgCol.reduceColumns(ee.Reducer.toList(), ["system:index"])
                        .get("list");
  indexList.evaluate(function(indexs) {
    for (var i=0; i<indexs.length; i++) {
      var image = imgCol.filter(ee.Filter.eq("system:index", indexs[i])).first();
      image = image.toInt16();
      Export.image.toDrive({
        image: image.clip(roi),
        description: indexs[i],
        fileNamePrefix: "Potain_1_"+indexs[i],
        region: roi,
        scale: 30,
        crs: "EPSG:4326",
        maxPixels: 1e13
      });
    }
  });
}
exportImageCollection(l5Imgs);
