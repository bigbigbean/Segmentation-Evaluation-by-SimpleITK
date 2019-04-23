# -*- coding: utf-8 -*-
 
import numpy as np
import os
import SimpleITK as sitk
import nibabel as nib
import pandas as pd

Hausdorff_list=list()
AvgHausdorff_list=list()
Dice_list=list()
Jaccard_list=list()  
Volume_list=list()  
False_negative_list=list()
False_positive_list=list()
mean_surface_dis_list=list()
median_surface_dis_list=list()
std_surface_dis_list=list()
max_surface_dis_list=list()

def file_name(file_dir):   
   L=[]   
   path_list = os.listdir(file_dir)
   path_list.sort() #对读取的路径进行排序
   for filename in path_list:
       if 'nii.gz' in filename:
        	L.append(os.path.join(filename))   
   return L 
 
def computeQualityMeasures(lP,lT):
    quality=dict()
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    #Hausdorff Distance
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()
    AvgHausdorff_list.append(quality["avgHausdorff"])
    Hausdorff_list.append(quality["Hausdorff"])
    #Dice,Jaccard,Volume Similarity..
    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["dice"]=dicecomputer.GetDiceCoefficient()
    quality["jaccard"]=dicecomputer.GetJaccardCoefficient()
    quality["volume_similarity"]=dicecomputer.GetVolumeSimilarity()
    quality["false_negative"]=dicecomputer.GetFalseNegativeError()
    quality["false_positive"]=dicecomputer.GetFalsePositiveError()
    Dice_list.append(quality["dice"])
    Jaccard_list.append(quality["jaccard"])  
    Volume_list.append(quality["volume_similarity"])  
    False_negative_list.append(quality["false_negative"])
    False_positive_list.append(quality["false_positive"])
    
    #Surface distance measures
    label = 1
    ref_distance_map=sitk.Abs(sitk.SignedMaurerDistanceMap(labelTrue>0.5,squaredDistance=False))
    ref_surface=sitk.LabelContour(labelTrue>0.5)
    statistics_image_filter=sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(labelTrue>0.5)
    num_ref_surface_pixels=int(statistics_image_filter.GetSum())

    seg_distance_map=sitk.Abs(sitk.SignedMaurerDistanceMap(labelPred>0.5,squaredDistance=False))
    seg_surface=sitk.LabelContour(labelPred>0.5)
    seg2ref_distance_map=ref_distance_map*sitk.Cast(seg_surface,sitk.sitkFloat32)
    ref2seg_distance_map=seg_distance_map*sitk.Cast(ref_surface,sitk.sitkFloat32)
    
    statistics_image_filter.Execute(labelPred>0.5)
    num_seg_surface_pixels=int(statistics_image_filter.GetSum())

    seg2ref_distance_map_arr=sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances=list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0])
    seg2ref_distances=seg2ref_distances+list(np.zeros(num_seg_surface_pixels-len(seg2ref_distances)))
    ref2seg_distance_map_arr=sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances=list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0])
    ref2seg_distances=ref2seg_distances+list(np.zeros(num_ref_surface_pixels-len(ref2seg_distances)))

    all_surface_distances=seg2ref_distances+ref2seg_distances
    quality["mean_surface_distance"]=np.mean(all_surface_distances)
    quality["median_surface_distance"]=np.median(all_surface_distances)
    quality["std_surface_distance"]=np.std(all_surface_distances)
    quality["max_surface_distance"]=np.max(all_surface_distances)
    mean_surface_dis_list.append(quality["mean_surface_distance"])
    median_surface_dis_list.append(quality["median_surface_distance"])
    std_surface_dis_list.append(quality["std_surface_distance"])
    max_surface_dis_list.append(quality["max_surface_distance"])

    return quality
 
gtpath = 'gt/'
predpath = 'pred/'
 
gtnames = file_name(gtpath)
prednames = file_name(predpath)

for i in range(len(gtnames)):
    # gt = sitk.ReadImage(gtpath + gtnames[i])
    img_path = os.path.join(gtpath + gtnames[i])
    img0 = nib.load(img_path)
    img = img0.get_data()
    seg_t = img.astype(np.float32)

    # pred = sitk.ReadImage(predpath + prednames[i])   
    img_path_1 = os.path.join(predpath + prednames[i])
    img1 = nib.load(img_path_1)
    img = img1.get_data()
    seg_st = img.astype(np.float32) 
    
    quality = computeQualityMeasures(seg_st,seg_t)
    print(gtnames[i])
    print(quality)
 
# img_path = os.path.join("Nifti-20190315162129646/20190315162129646-seg-2.nii.gz")
# img0 = nib.load(img_path)
# img = img0.get_data()
# seg_t = img.astype(np.float32)

# img_path_1 = os.path.join("Nifti-20190315162129646/20190315162129646-seg-st-2.nii.gz")
# img1 = nib.load(img_path_1)
# img = img1.get_data()
# seg_st = img.astype(np.float32)
 
# quality = computeQualityMeasures(seg_st,seg_t)
# print(quality) 
data_frame1 = pd.DataFrame({'filename':gtnames,'Dice':Dice_list,'Jaccard':Jaccard_list,
  'False Negative':False_negative_list,'False Positive':False_positive_list,
  'Hausdorff Distance': Hausdorff_list,'avgHausdorff Distance':AvgHausdorff_list,
  'Mean Surface Distance':mean_surface_dis_list,'Median Surface Distance':median_surface_dis_list,
  'Std Surface Distance':std_surface_dis_list,'Max Surface Distance':max_surface_dis_list})
data_frame1.to_csv("quality.csv",index=False)

 
