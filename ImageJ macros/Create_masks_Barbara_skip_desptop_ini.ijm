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
// ***ALSO MUST OPEN AN IMAGE OF THE CORRECT SIZE WHICH NAME MATCHES LINE #96
count = 0;

list = getFileList(dir);
for (i=0; i<list.length; i++) {
     count++;
	 print(list[i]);
}
n = 0;
//processFiles(dir);
print(count + "files processed");

add_color = 1;
last_num_roi = 0;
for (i = 1; i < list.length; i++) {  // CHANGE 1 ==> if currently 1, that means skips desktop.ini

	path = dir + list[i];
	print(path);
	//open(path);
	match = 0;
	if (i == 1) {  // CHANGE 1
		match = 0;
	}
	else {
		cur_f = split(list[i - 1], "_");
		next_f = split(list[i], "_");

		if (cur_f[0] + cur_f[cur_f.length - 2] == next_f[0] + next_f[next_f.length - 2]) {
			match = 1;
			print(next_f[0] + next_f[next_f.length - 1]);
			print(cur_f[0] + cur_f[cur_f.length - 1]);
			print("Im in");
		}
	}

	// if they do match, then just open and append the ROI without saving the image
	if (match == 1) {
		roiManager("Open", path);
		add_color = add_color;
	}
	else if (match == 0 && i == 1) {   // CHANGE 1
		newImage("Labeling", "8-bit black", getWidth(), getHeight(), 1);		
		roiManager("Open", path);
	}
	
	else if (match == 0) {
		add_color = 1;
		// save image
		selectWindow("Labeling");
		setThreshold(0, 0);
		run("Convert to Mask");
		print(dir + "Mask" + path);
		tmpStr = substring(list[i - 1], 0, lengthOf(list[i - 1]) - 6);
		sav_Name = tmpStr + "_pos_truth.tif";
		saveAs("Tiff", dir + sav_Name);	
		close();		

	    // then delete everything in the ROI manager
		for (index = 0; index < roiManager("count"); index++) {
			roiManager("delete");
			print(index);
		}
		roiManager("Open", path);
		newImage("Labeling", "8-bit black", getWidth(), getHeight(), 1);
		
	}
	selectWindow("Labeling");

	// print with different indices
	index = 0;
	last_num_roi = roiManager("count");

	while (index < roiManager("count")) {
		roiManager("select", index);
		setColor(add_color);
		fill();
		index++;
	}
	
	resetMinAndMax();
	run("glasbey");
	

	
	selectWindow("EAE_miR_4_jaune_Series006_z000.tif");  // IMAGE MUST BE CORRECT SIZE

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

// save image
selectWindow("Labeling");
setThreshold(0, 0);
run("Convert to Mask");
print(dir + "Mask" + path);
tmpStr = substring(list[i - 1], 0, lengthOf(list[i - 1]) - 6);
sav_Name = tmpStr + "_pos_truth.tif";
saveAs("Tiff", dir + sav_Name);	
close();	