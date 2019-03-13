// delete everything in ROImanager
for (index = 0; index < roiManager("count"); index++) {
	roiManager("delete");
	print(index);
}

// read in files to "filesDir"
//dir = getDirectory("Choose a Directory");
//dir = "J:\\DATA_2017-2018\\Optic_nerve\\EAE_miR_AAV2\\2018.08.07\\ON_11\\ROIs\\"
dir = "C:\\Users\\Neuroimmunology Unit\\Documents\\GitHub\\Optic Nerve\\Etienne\\ROIs\\"
//setBatchMode(true);


// ***ALSO MUST OPEN AN IMAGE OF THE CORRECT SIZE WHICH NAME MATCHES LINE #48

count = 0;

list = getFileList(dir);
for (i=0; i<list.length; i++) {
     count++;
	 print(list[i]);
}
n = 0;
//processFiles(dir);
print(count + "files processed");

for (i = 0; i < list.length; i++) {

	path = dir + list[i];
	print(path);
    roiManager("Open", path);
	roiManager("Deselect");
	roiManager("Combine");
	run("Create Mask");
	run("Invert");
    selectWindow("Mask");
    tmpStr = substring(list[i], 0, lengthOf(list[i]) - 6);
	sav_Name = tmpStr + "_pos_truth.tif";
	saveAs("Tiff", dir + sav_Name);	
	close();		

	// then delete everything in the ROI manager
	for (index = 0; index < roiManager("count"); index++) {
			roiManager("delete");
			print(index);
	}
	
	selectWindow("EAE_miR_4_jaune_Series006_z000.tif"); // IMAGE MUST BE CORRECT SIZE

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

}