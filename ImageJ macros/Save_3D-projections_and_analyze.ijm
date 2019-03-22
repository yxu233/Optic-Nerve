// read in files to "filesDir"
dir = getDirectory("Choose a Directory");
//dir = "J:\\DATA_2017-2018\\Optic_nerve\\EAE_miR_AAV2\\2018.08.07\\ON_11\\ROIs\\"
//dir = "C:\\Users\\Neuroimmunology Unit\\Documents\\GitHub\\Optic Nerve\\Etienne\\Control Images\\"
//dir = "C:\\Users\\Neuroimmunology Unit\\Documents\\GitHub\\Optic Nerve\\Training Data\\New folder\\"

//setBatchMode(true);
// ***ALSO MUST OPEN AN IMAGE OF THE CORRECT SIZE WHICH NAME MATCHES LINE #96

input_file_num = 0;
bleb_file_num = 5;


list = getFileList(dir);
count = 0;
for (i=0; i<list.length; i++) {
     count++;
	 print(list[i]);
}
n = 0;
//processFiles(dir);
print(count + "files processed");

//add_color = 1;
//last_num_roi = 0;
//for (i = 0; i < list.length; i++) {

// (1) opens INPUT image
path = dir + list[input_file_num];
print(path);
open(path); 
//selectWindow("1) EAE_miR_11_jaune_Series001_z_input_stack.tif");
run("Green");

// (2) opens BLEBS image
path = dir + list[bleb_file_num];
print(path);
open(path); 
//selectWindow("6) EAE_miR_11_jaune_Series001_z_DISTANCE_THRESHED_post-processed.tif");
run("Red");

// (3) Merge to create 3D projects
//run("Merge Channels...", "c1=[1) EAE_miR_11_jaune_Series001_z_input_stack.tif] c2=[6) EAE_miR_11_jaune_Series001_z_DISTANCE_THRESHED_post-processed.tif] create keep");
run("Merge Channels...", "c1=[" + list[input_file_num] + "] c2=[" + list[bleb_file_num] + "] create keep");
tmpStr = substring(list[input_file_num], 0, lengthOf(list[input_file_num]) - 4);
sav_Name = tmpStr + "_composite.tif";
saveAs("Tiff", dir + "/ImageJ outputs/" + sav_Name);
run("3D Project...", "projection=[Brightest Point] axis=Y-Axis slice=1 initial=0 total=360 rotation=10 lower=1 upper=255 opacity=0 surface=100 interior=50");
//saveAs("Tiff", "C:/Users/Neuroimmunology Unit/Documents/GitHub/Optic Nerve/Results_ON_11_Checkpoint_400000/ImageJ outputs/1) EAE_miR_11_jaune_Series001_composite-PROJECTION.tif");
sav_Name = tmpStr + "_composite-PROJECTION.tif";
saveAs("Tiff", dir + "/ImageJ outputs/" + sav_Name);


selectWindow(list[bleb_file_num]);
run("3D Objects Counter", "threshold=1 slice=80 min.=10 max.=168820736 exclude_objects_on_edges surfaces statistics summary");
//Table.rename("Statistics for C2-1) EAE_miR_11_jaune_Series001_analysis.csv", "Statistics for C2-1) EAE_miR_11_jaune_Series001_analysis.csv");
//saveAs("Results", "C:/Users/Neuroimmunology Unit/Documents/GitHub/Optic Nerve/Results_ON_11_Checkpoint_400000/ImageJ outputs/Statistics for C2-1) EAE_miR_11_jaune_Series001_analysis.csv");
sav_Name = tmpStr + "_analysis.csv";
saveAs("Results", dir + "/ImageJ outputs/" + sav_Name);


//selectWindow("Surface map of C2-1) EAE_miR_11_jaune_Series001_composite.tif");
//saveAs("Tiff", "C:/Users/Neuroimmunology Unit/Documents/GitHub/Optic Nerve/Results_ON_11_Checkpoint_400000/ImageJ outputs/1) EAE_miR_11_jaune_Series001_surfaces.tif");
sav_Name = tmpStr + "_surface_map.tif";
saveAs("Tiff", dir + "/ImageJ outputs/" + sav_Name);

run("3D Project...", "projection=[Brightest Point] axis=Y-Axis slice=1 initial=0 total=360 rotation=10 lower=1 upper=255 opacity=0 surface=100 interior=50");
//saveAs("Tiff", "C:/Users/Neuroimmunology Unit/Documents/GitHub/Optic Nerve/Results_ON_11_Checkpoint_400000/ImageJ outputs/1) EAE_miR_11_jaune_Series001_3D-project_surfaces.tif");
sav_Name = tmpStr + "_surface_map-PROJECTION.tif";
saveAs("Tiff", dir + "/ImageJ outputs/" + sav_Name);


run("Close All");
call("java.lang.System.gc");    // clears memory leak
call("java.lang.System.gc"); 
call("java.lang.System.gc"); 
call("java.lang.System.gc"); 
call("java.lang.System.gc"); 
call("java.lang.System.gc"); 
call("java.lang.System.gc"); 
call("java.lang.System.gc"); 
call("java.lang.System.gc"); 
call("java.lang.System.gc"); 

