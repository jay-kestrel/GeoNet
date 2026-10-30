import numpy as np
import time
from scipy import ndimage
from pygeonet_rasterio import *
from pygeonet_vectorio import *
from pygeonet_plot import *
from numba import jit
import pygeonet_kernels

def select_skeleton_size_threshold(skeletonNumElementsHistogramX):
    """Component-size threshold from the size histogram's bin edges.

    Picks the first edge above the edges' 30th percentile (upstream
    behavior). When that strict selection is empty - all-equal edges -
    falls back to the quantile edge itself so the threshold is always
    defined (review probe P3: the old inline code raised IndexError).
    """
    test_thresh = np.where(skeletonNumElementsHistogramX>np.quantile(skeletonNumElementsHistogramX,0.3))[0]
    if test_thresh.size == 0:
        return float(np.quantile(skeletonNumElementsHistogramX, 0.3))
    return float(skeletonNumElementsHistogramX[test_thresh[0]])


def _find_skeleton_end_points(skeletonFromFlowAndCurvatureArray,
                              skeletonLabeledArray,
                              skeletonNumConnectedComponentsList,
                              geodesicDistanceArray):
    """Component-size statistics plus the end-point scan.

    Returns the [row, col] channel-head candidate list; empty when no
    component qualifies.
    """
    lbls = np.arange(1, skeletonNumConnectedComponentsList+1)

    skeletonLabeledArrayNumtuple = ndimage.labeled_comprehension(skeletonFromFlowAndCurvatureArray,\
                                                                 skeletonLabeledArray,\
                                                                 lbls,np.count_nonzero,\
                                                                 int,0)
    skeletonNumElementsSortedList = np.sort(skeletonLabeledArrayNumtuple)
    #histarray,skeletonNumElementsHistogramX=np.histogram(\
    #    skeletonNumElementsSortedList[0:len(skeletonNumElementsSortedList)-1],
    #    int(np.floor(np.sqrt(len(skeletonNumElementsSortedList)))))

    histarray,skeletonNumElementsHistogramX=np.histogram(\
        skeletonNumElementsSortedList,bins='auto')
    print(f'Max: {np.max(skeletonNumElementsHistogramX)}')
    print(f'Median: {np.median(skeletonNumElementsHistogramX)}')

    #if defaults.doPlot == 1:
    #    raster_plot(skeletonLabeledArray, 'Skeleton Labeled Array elements Array')
    # Create skeleton gridded array
    skeleton_label_set, label_indices = np.unique(skeletonLabeledArray, return_inverse=True)
    skeletonNumElementsGriddedArray = np.array([skeletonLabeledArrayNumtuple[x-1] for x in skeleton_label_set])[label_indices].reshape(skeletonLabeledArray.shape)
    #if defaults.doPlot == 1:
    #    raster_plot(skeletonNumElementsGriddedArray,
    #                'Skeleton Num elements Array')

    # Elements smaller than skeletonNumElementsThreshold are not considered in the
    # skeletonEndPointsList detection

    skeletonNumElementsThreshold = select_skeleton_size_threshold(skeletonNumElementsHistogramX)
    print(f'skeletonNumElementsThreshold: {str(skeletonNumElementsThreshold)}')

    # Scan the array for finding the channel heads
    skeletonEndPointsList = []
    nrows = skeletonFromFlowAndCurvatureArray.shape[0]
    ncols = skeletonFromFlowAndCurvatureArray.shape[1]
    search_box = defaults.endPointSearchBoxSize
    for i in range(nrows):
        for j in range(ncols):
            if skeletonLabeledArray[i,j]!=0 \
               and skeletonNumElementsGriddedArray[i,j]>=skeletonNumElementsThreshold:
                # Define search box and ensure it fits within the DTM bounds
                my = i-1
                py = nrows-i
                mx = j-1
                px = ncols-j
                xMinus = np.min([search_box, mx])
                xPlus  = np.min([search_box, px])
                yMinus = np.min([search_box, my])
                yPlus  = np.min([search_box, py])
                # Extract the geodesic distances geodesicDistanceArray for pixels within the search box
                searchGeodesicDistanceBox = geodesicDistanceArray[i-yMinus:i+yPlus, j-xMinus:j+xPlus]

                # Extract the skeleton labels for pixels within the search box
                searchLabeledSkeletonBox = skeletonLabeledArray[i-yMinus:i+yPlus, j-xMinus:j+xPlus]
                # Look in the search box for skeleton points with the same label
                # and greater geodesic distance than the current pixel at (i,j)
                # - if there are none, then add the current point as a channel head
                v = searchLabeledSkeletonBox==skeletonLabeledArray[i,j]
                v1 = v * searchGeodesicDistanceBox > geodesicDistanceArray[i,j]
                v3 = np.where(np.any(v1==True,axis=0))
                if len(v3[0])==0:
                    skeletonEndPointsList.append([i,j])
        if i%500==0:
            print(f'Iterations Remaining: {nrows-i}')
    #skeletonEndPointsList = skeleton_endPoint_loop(nrows,ncols,skeletonLabeledArray,skeletonNumElementsGriddedArray,skeletonNumElementsThreshold,geodesicDistanceArray,search_box)
    return skeletonEndPointsList


def Channel_Head_Definition(skeletonFromFlowAndCurvatureArray, geodesicDistanceArray):
    # Locating end points
    print("Locating skeleton end points")
    structure = np.ones((3, 3))
    skeletonLabeledArray, skeletonNumConnectedComponentsList =\
                          ndimage.label(skeletonFromFlowAndCurvatureArray,
                                        structure=structure)
    """
     Through the histogram of skeletonNumElementsSortedList
     (skeletonNumElementsList minus the maximum value which
      corresponds to the largest connected element of the skeleton) we get the
      size of the smallest elements of the skeleton, which will likely
      correspond to small isolated convergent areas. These elements will be
      excluded from the search of end points.
    """
    print("Counting the number of elements of each connected component")
    if skeletonNumConnectedComponentsList == 0:
        # Degenerate skeleton: no components, so zero channel heads; the
        # component-size statistics are undefined here.
        print('Skeleton has no connected components; no channel heads to find')
        skeletonEndPointsList = []
    else:
        skeletonEndPointsList = _find_skeleton_end_points(
            skeletonFromFlowAndCurvatureArray, skeletonLabeledArray,
            skeletonNumConnectedComponentsList, geodesicDistanceArray)
    if len(skeletonEndPointsList) == 0:
        # Zero heads: nothing qualified (or no components). Produce defined
        # empty outputs instead of an IndexError on the empty transpose
        # (review probe P4).
        print('No channel heads found; writing empty channel-head outputs')
        skeletonEndPointsListArray = np.zeros((2, 0), dtype=int)
    else:
        skeletonEndPointsListArray = np.transpose(skeletonEndPointsList)

    if defaults.doPlot == 1 and skeletonEndPointsListArray.shape[1] > 0:
        raster_point_plot(skeletonFromFlowAndCurvatureArray, skeletonEndPointsListArray,
                          'Skeleton Num elements Array with channel heads', cm.binary, 'ro')
    if defaults.doPlot == 1 and skeletonEndPointsListArray.shape[1] > 0:
        raster_point_plot(geodesicDistanceArray, skeletonEndPointsListArray,
                          'Geodesic distance Array with channel heads', cm.coolwarm, 'ro')
    xx = skeletonEndPointsListArray[1]
    yy = skeletonEndPointsListArray[0]
    # Write shapefiles of channel heads
    write_drainage_nodes(xx,yy,"ChannelHead",
                         Parameters.pointFileName,Parameters.pointshapefileName)
    # Write raster of channel heads
    channelheadArray = np.zeros((geodesicDistanceArray.shape))
    channelheadArray[skeletonEndPointsListArray[0],
                     skeletonEndPointsListArray[1]] = 1
    outfilepath = Parameters.geonetResultsDir
    demName = Parameters.demFileName
    outfilename = demName.split('.')[0]+'_channelHeads.tif'
    write_geotif_generic(channelheadArray,\
                         outfilepath,outfilename)
    return xx, yy
def main():
    outfilepath = Parameters.geonetResultsDir
    demName = Parameters.demFileName.split('.')[0]
    skeleton_filename = demName+'_skeleton.tif'
    skeletonFromFlowAndCurvatureArray = read_geotif_generic(outfilepath, skeleton_filename)[0]
    geodesic_filename = demName+'_geodesicDistance.tif'
    geodesicDistanceArray = read_geotif_generic(outfilepath, geodesic_filename)[0]
    if pygeonet_kernels._native_module() is None:
        # Python fallback: Channel_Head_Definition keeps its own plotting and
        # raster/vector writes.
        pygeonet_kernels.find_channel_heads(skeletonFromFlowAndCurvatureArray,
                                            geodesicDistanceArray)
    else:
        # Native kernel is compute-only; plotting and raster/vector writes
        # stay Python-side, mirroring the original function's outputs.
        xx, yy = pygeonet_kernels.find_channel_heads(
            skeletonFromFlowAndCurvatureArray, geodesicDistanceArray)
        _write_channel_head_outputs(skeletonFromFlowAndCurvatureArray,
                                    geodesicDistanceArray, xx, yy)


def _write_channel_head_outputs(skeletonFromFlowAndCurvatureArray,
                                geodesicDistanceArray, xx, yy):
    """Native path: the output half of the original function, Python-side."""
    skeletonEndPointsListArray = np.stack([yy, xx])
    if defaults.doPlot == 1:
        raster_point_plot(skeletonFromFlowAndCurvatureArray, skeletonEndPointsListArray,
                          'Skeleton Num elements Array with channel heads', cm.binary, 'ro')
    if defaults.doPlot == 1:
        raster_point_plot(geodesicDistanceArray, skeletonEndPointsListArray,
                          'Geodesic distance Array with channel heads', cm.coolwarm, 'ro')
    # Write shapefiles of channel heads
    write_drainage_nodes(xx,yy,"ChannelHead",
                         Parameters.pointFileName,Parameters.pointshapefileName)
    # Write raster of channel heads
    channelheadArray = np.zeros((geodesicDistanceArray.shape))
    channelheadArray[yy, xx] = 1
    outfilepath = Parameters.geonetResultsDir
    demName = Parameters.demFileName
    outfilename = demName.split('.')[0]+'_channelHeads.tif'
    write_geotif_generic(channelheadArray,\
                         outfilepath,outfilename)





if __name__ == '__main__':
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    print(f'time taken to complete channel head definition: {t1-t0} seconds')