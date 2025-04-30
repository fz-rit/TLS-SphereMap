#!/usr/bin/env python
"""
Importer to convert a gbl file (raw output from a Compact Biomass Lidar (CBL) scan to text format and Sorted Pulse Data (SPD) format (if spdlib is available).

Written by Jasmine Muir (Queensland Department of Science, IT, and Innovation; University of Queensland) 
and Ian Paynter (University of Massachusetts - Boston)

March 2015
"""

from __future__ import print_function
import sys,os
import argparse
import re
import numpy as np
from pathlib import Path
try:
    import spdpy
    hasSpd = True
except ImportError:
    hasSpd = False
    
#Set global variables
maxEncoderValue = int("3FFF",16)


def getCmdargs():
    """
    Get command-line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--inFolder", type=str, help="Input folder containing GBL files (Required)")
    p.add_argument("--cblVersion", type=int, default=2, help="Options: 1 | 2")
    p.add_argument("--agh", default=1.2, help="Above ground height of sensor optical centre")
    p.add_argument("--verbose", "-v", default=False, action="store_true", help="Verbose. Default False.")

    cmdargs = p.parse_args()

    # Check if inFolder is provided
    if cmdargs.inFolder is None:
        p.print_help()
        sys.exit()

    # Convert folder path to Path object
    in_folder_path = Path(cmdargs.inFolder)

    # Check if folder exists
    if not in_folder_path.exists() or not in_folder_path.is_dir():
        print(f"Error: The folder '{in_folder_path}' does not exist or is not a directory.")
        sys.exit()

    # Collect all files inside the folder
    cmdargs.inFiles = sorted([str(file) for file in in_folder_path.glob('*') if file.is_file()])

    if not cmdargs.inFiles:
        print(f"Error: No files found in '{in_folder_path}'.")
        sys.exit()

    return cmdargs
    
    
def get_ScanInfo(inFile):
    """
    Determine the maximum scanLine ID and other scan information
    """
    
    nPulses = 0
    nPoints = 0
    maxScanLineID = 0
    noScanlines = 0 
    
    ascObj = open(inFile, 'r')
    contents = ascObj.read()
    p = re.compile('\x03')
    contents = p.split(contents)
    
    for scanline in contents:      
        c = re.compile('DIST1|RSSI1|DIST2|RSSI2')
        scanlineparts = c.split(scanline)
        if len(scanlineparts) == 5:
            noScanlines+=1
            DIST1 = scanlineparts[1].strip().split(" ")
            DIST2 = scanlineparts[2].strip().split(" ")
            return1Range = DIST1[5:]
            return2Range = DIST2[5:]          
            maxScanLineID = max(maxScanLineID, len(return1Range))
            nPulses += len(return1Range) 
            return2RangeValid = len([i for i in return2Range if int(i,16) > 0])
            return1RangeValid = len(return1Range)           
            nPoints += (return2RangeValid + return1RangeValid)                
    return noScanlines, maxScanLineID, nPulses, nPoints

def writeTxtFile(txtFile,pulseDict,numOfReturns):   
    """
    Write out the text file line for first returns
    """
    returnNo = 1
    outline = "%s,%s,%s,%s,%s,%s,%s,%s\r\n" %(
        pulseDict["return1xyz"][0],
        pulseDict["return1xyz"][1],
        pulseDict["return1xyz"][2],
        np.degrees(pulseDict["zenith"]),
        np.degrees(pulseDict["azimuth"]),
        pulseDict["range1metres"],
        pulseDict["intensity1"],
        returnNo)
    txtFile.write(outline)                

    if numOfReturns == 2:
        #Write out the text file line for second returns
        returnNo = 2
        outline = "%s,%s,%s,%s,%s,%s,%s,%s\r\n" %(
        pulseDict["return2xyz"][0],
        pulseDict["return2xyz"][1],
        pulseDict["return2xyz"][2],
        np.degrees(pulseDict["zenith"]),
        np.degrees(pulseDict["azimuth"]),
        pulseDict["range2metres"],        
        pulseDict["intensity2"],
        returnNo)
        txtFile.write(outline)

def spherical2cartesian(zenith,azimuth,r):
    """
    Converts spherical to cartesian coordinates
    phi = zenith theta =azimuth r = radial distance
    http://mathworld.wolfram.com/SphericalCoordinates.html
    """
    x = -(r * np.sin(zenith) * np.cos(azimuth))
    y = r * np.sin(zenith) * np.sin(azimuth)
    z = r * np.cos(zenith)
    return (x,y,z) 
    
def findSPDminMax(x,y,z,xMin,xMax,yMin,yMax,zMin,zMax):
    """
    Test if the x,y,z excede current global scan minimum and maximum values
    """     

    xMin = min(x,xMin)
    xMax = max(x,xMax)
    yMin = min(y,yMin)
    yMax = max(y,yMax)
    zMin = min(z,zMin)
    zMax = max(z,zMax)
    
    return xMin,xMax,yMin,yMax,zMin,zMax
    
def readScan(inFile):
    """
    Open and read gbl file
    """
    
    ascObj = open(inFile, 'r')
    contents = ascObj.read()
    p = re.compile('\x03')
    scanlines = p.split(contents)   
    return scanlines
    
    
def readScanline(scanlineparts):
    """
    Split each scanline into it's parts
    """
    scanlineDict = {}
    scanlineHeader = scanlineparts[0].strip().split(" ")
    DIST1 = scanlineparts[1].strip().split(" ")
    DIST2 = scanlineparts[2].strip().split(" ")
    RSSI1 = scanlineparts[3].strip().split(" ")            
    RSSI2 = scanlineparts[4].strip().split(" ")

    scanlineDict["return1Range"] = DIST1[5:]
    scanlineDict["return1Intensity"] = RSSI1[5:]
    scanlineDict["return2Range"] = DIST2[5:]
    scanlineDict["return2Intensity"] = RSSI2[5:]
    angularWidth = int(DIST1[3],16)/10000.0

    #assumes scanner rotating 270 degrees in zenith plane
    scanlineDict["numberofZenithReturns"] = int(270/angularWidth)
    scanlineDict["scanlineError1"] = scanlineHeader[5]
    scanlineDict["scanlineError2"] = scanlineHeader[6]
    #print(int(scanlineHeader[   
    #scanlineDict["azimuthPosition"] = int(scanlineHeader[19],16)*0.004
    scanlineDict["azimuthPosition"] = int(scanlineHeader[19],16)

    return scanlineDict      
    
    
         
def readPulse(i,scanlineDict,zenithSpacing,scanlineMidPulse,maxScanLineID):        

    """
    Read in each pulse. "i" is the counter for the pulse index in each scanline read in.
    Note that bothe CBL1 and CBL2 rotate in a counter-clockwise direction. Because of this the azimuth rotation
    must be converted to clockwise and split at zenith nadir - the mid pulse for each scanline(scanlineMidPulse) 
    to match the spd file format definition of 0-360 degree azimuth (converted to radians).
    Individual zenith increments are not recorded - they are preset in the SICK lidar to either:
        - 0.25 degrees (CBL2) or 
        - 0.50 degrees(CBL1).  
    """
    pulseDict = {}

    range1 = int(scanlineDict["return1Range"][i],16)
    range2 = int(scanlineDict["return2Range"][i],16)
    pulseDict["intensity1"] = int(scanlineDict["return1Intensity"][i],16)
    pulseDict["intensity2"] = int(scanlineDict["return2Intensity"][i],16)

    #convert range from mm to metres
    pulseDict["range1metres"] = range1/1000.0
    pulseDict["range2metres"] = range2/1000.0
    
    #Determine the azimuth adjustment  to use acounting for scanner azimuth rotation during each zenith mirror rotation
    #maxScanLineID = number of pulses per scanline
    
    azimuthAdjustment =  scanlineDict["encoderPositionAzimuthDiff"]/maxScanLineID

    #Determine pulse azimuth and zenith
    
    if i < scanlineMidPulse: 
         azimuth = 360- (scanlineDict["azimuthPosition"] + (azimuthAdjustment*i))
         pulseDict["zenith"] = np.radians((scanlineMidPulse-i) * zenithSpacing)         
    else:
         azimuth = 180- (scanlineDict["azimuthPosition"]+(azimuthAdjustment*i))
         pulseDict["zenith"] = np.radians((i-scanlineMidPulse) * zenithSpacing)
                
    #return azimuth to positive value in the case it goes past 360 or below zero
    if azimuth >= 360:
        azimuth = azimuth - 360
    elif azimuth < 0:
        azimuth =  360 + azimuth
           
    pulseDict["azimuth"] = np.radians(azimuth)
    
    #Use a function to determine the x,y,z cartesian co-ordinates from azimuth and zenith
    pulseDict["return1xyz"] = spherical2cartesian(pulseDict["zenith"],pulseDict["azimuth"],pulseDict["range1metres"])
    pulseDict["return2xyz"] = spherical2cartesian(pulseDict["zenith"],pulseDict["azimuth"],pulseDict["range2metres"])
          
    return pulseDict
                      
def findEncoderDiff(scanlines):
    """
    The encoder value has a maximum value of 16383 (Hexidecimal "3FFF" base 16) and is reset after this value to zero.
    We need to determine the absolute increment of encoder ticks including the part from previous value to 
    maximum encoder value, plus the increment from zero to the current encoder value. The total encoder position change
    between each scanline is written to the array encoderPositionsDiff
    """
    
    #First initialise a list to store the encoder positions (ticks)
    encoderPositions = []
    
    #read in the scanlines and create a dictionary of their parts. Append the encoder positions to a seperate list (encoderPositions)
    for scanline in scanlines:
        c = re.compile('DIST1|RSSI1|DIST2|RSSI2')
        scanlineparts = c.split(scanline)
        if len(scanlineparts) == 5:
            scanlineDict = readScanline(scanlineparts)
            encoderPositions.append(scanlineDict["azimuthPosition"])
            
    encoderPositions = np.array(encoderPositions)
    
    #Initialise array to hold values for total encoder position change (difference)
    encoderPositionsDiff = np.zeros(np.size(encoderPositions))
    
    #Loop through the encoder positions. Where the current position is greater then the previous position the encoder has been reset.
    #In this case we add the difference between the maximum encoder position and the previous encoder position, plus the current encoder position
    #to find the total encoder psotion change. In other cases total encoder position change is the difference between the current and previous position.
    for m in range(np.size(encoderPositions)):
        #Don't do this if it's the first position in the array (first scanline)
        if m > 0 and (m < (np.size(encoderPositions)-5)):
            if encoderPositions[m]<encoderPositions[m-1]: 
                encoderPositionsDiff[m] = maxEncoderValue-encoderPositions[m-1]+encoderPositions[m]
            else:    
                encoderPositionsDiff[m] = encoderPositions[m] - encoderPositions[m-1]
        #Account for annomoulous behaviour at end of scan including reported encoder values corresponding to the platform moving backwards.
        elif m >= (np.size(encoderPositions)-5):
            if encoderPositions[m]<encoderPositions[m-1]: 
                tempEncoderPositionsDiff = encoderPositions[m]-encoderPositions[m-1]
                if tempEncoderPositionsDiff < 0:
                     encoderPositionsDiff[m] = encoderPositionsDiff[m-1]
                elif (tempEncoderPositionsDiff >= 0): 
                    encoderPositionsDiff[m] = maxEncoderValue-encoderPositions[m-1]+encoderPositions[m]      
            else:    
                encoderPositionsDiff[m] = encoderPositions[m] - encoderPositions[m-1]        
    
    return encoderPositionsDiff            
    
def doImport(inFile, cblVersion, agh, verbose):
    """
    Main routine - writes out text file, ply, spd and las file
    """
    
    #Set the instrument dependant variables for CBL1 and CBL2
    if cblVersion == 1:
        print("CBL Version %s" %cblVersion)
        encoderToAzimuthCorrectionFactor = 1/250.0
        zenithSpacing = 0.5
        azimuthSpacing = 0.25
        scanlineMidPulse = 270
    elif cblVersion == 2:
        print("CBL Version %s" %cblVersion)
        encoderToAzimuthCorrectionFactor = 1/250.0
        zenithSpacing = 0.25
        azimuthSpacing = 0.25
        scanlineMidPulse = 540
    else:
        print("CBL Version %s" %cblVersion)
        sys.exit("You're getting ahead of the technology!! There are only CBL1 and CBL2 instruments so far...")    
    
    #Set file paths
    # Create the Text file
    outputFileTxt = inFile.replace(".gbl",".txt")
    # outputFilePly = inFile.replace(".gbl",".ply")
    if os.path.exists(outputFileTxt):
        os.remove(outputFileTxt)
    # if os.path.exists(outputFilePly):
      #  os.remove(outputFilePly)    
    txtFile = open(outputFileTxt,'w')
    # plyFile = open(outputFilePly,'w')
    
    # Get scan info
    (noScanlines, maxScanLineID, nPulses, nPoints) = get_ScanInfo(inFile)
    
    # Create the ply file header
    #writePlyHeader(plyFile,nPoints)

    pointsStartIdx = 0
    scanlineNumber = 0
    pulseID = 0
    firstScanLine = True
    firstPulse = True
    zeroReturnCount = 0
      
    # Create the SPD file
    if hasSpd:
        outputFileSpd = inFile.replace(".gbl",".spd")
        spdOutFile = spdpy.createSPDFile(outputFileSpd)
        spdWriter = spdpy.SPDPySeqWriter()
        spdOutFile.setNumBinsY(noScanlines)
        spdOutFile.setNumBinsX(maxScanLineID)
        spdOutFile.setBinSize(1)
        spdOutFile.setIndexType(5)
        spdWriter.open(spdOutFile,outputFileSpd)

        # Define contents of this TLS dataset
        spdOutFile.setReceiveWaveformDefined(0)
        spdOutFile.setTransWaveformDefined(0)
        spdOutFile.setDecomposedPtDefined(0)
        spdOutFile.setDiscretePtDefined(1)
        spdOutFile.setOriginDefined(0)
        spdOutFile.setHeightDefined(0)
        spdOutFile.setRgbDefined(0)
        spdOutFile.setPulseAngularSpacingAzimuth(np.radians(azimuthSpacing))
        spdOutFile.setPulseAngularSpacingZenith(np.radians(zenithSpacing))
        spdOutFile.setBeamDivergence(15.0)

        # Define scanner properties
        spdOutFile.setSensorHeight(agh)
        spdpy.setSPDFileWavelengthsAndBandwidths(spdOutFile,[905.0],[1])
    


        #Initialiase spd file min/max for x,y,z    
        xMin = 0
        xMax = 0
        yMin = 0
        yMax = 0
        zMin = 0
        zMax = 0

        spdOutFile.setScanlineMin(scanlineNumber)
        spdOutFile.setScanlineMax(noScanlines)
    

    scanlines = readScan(inFile)        
    encoderPositionsDiff = findEncoderDiff(scanlines)  

    j = 0
    totalAzimuthRotation = 0
    
    
    #print(encoderPositionsDiff)             
    for scanline in scanlines:
        c = re.compile('DIST1|RSSI1|DIST2|RSSI2')
        scanlineparts = c.split(scanline)

        if len(scanlineparts) == 5:
            scanlineDict = readScanline(scanlineparts)
            totalAzimuthRotation += encoderPositionsDiff[j]*encoderToAzimuthCorrectionFactor
            scanlineDict["azimuthPosition"] = totalAzimuthRotation
            scanlineDict["encoderPositionAzimuthDiff"] = encoderPositionsDiff[j]*encoderToAzimuthCorrectionFactor           
            scanlineIndex = 0
            if hasSpd:
                spdOutFile.setScanlineIdxMin(scanlineIndex)
                if firstScanLine:
                    firstScanLine = False
                else:               
                    spdWriter.writeDataRow(scanLineData, scanlineNumber-1)
                    if verbose:
                        perCompleted = int((float(scanlineNumber)/noScanlines)*100.0)
                        print("Percentage completed", perCompleted, end="\r")
           
            scanLineData = [[] for i in range(maxScanLineID)]
            scanlineNumber+=1     
            
            for i in range(maxScanLineID):        
                pulseDict = readPulse(i,scanlineDict,zenithSpacing,scanlineMidPulse,maxScanLineID)
                
                if pulseDict["range2metres"] > 0:
                    numOfReturns = 2
                elif pulseDict["range2metres"] == 0 and pulseDict["range1metres"] > 0:
                    numOfReturns = 1
                elif pulseDict["range2metres"] == 0 and pulseDict["range1metres"] == 0:
                    numOfReturns = 0
                    zeroReturnCount += 1
                else:
                    print(pulseDict["range2metres"], pulseDict["range1metres"])
                    sys.exit("some weird thing happening with number of returns")           
 
                #Write textfile Lines and return xyz values for first and second returns
                writeTxtFile(txtFile,pulseDict,numOfReturns)
                #writePlyFile(plyFile,pulseDict,numOfReturns)

                ## ------------------------------------------
                ## below requires spd
                ## -----------------------------------------
                if hasSpd:
                              
                    #Create an spd pulse
                    if firstPulse:
                        firstPulse = False
                    else:                    
                        scanLineData[pulse.scanlineIdx].append(pulse)

                    pulse = spdpy.createSPDPulsePy()
                    pulse.numberOfReturns = numOfReturns
                    if numOfReturns != 0:
                        pulse.pts_start_idx = pointsStartIdx
                    else:
                        pulse.pts_start_idx = -1    
                    pulse.wavelength = float(905.0)
                    pulse.pulseID = pulseID

                    #check errors and set in spdfile
                    if int(scanlineDict["scanlineError1"]) != 1:
                        pulse.user = 1
                    elif int(scanlineDict["scanlineError2"]) != 1:
                        pulse.user = 2    
                    pulse.azimuth = float(pulseDict["azimuth"])
                    pulse.zenith = float(pulseDict["zenith"])
                    pulse.scanline = scanlineNumber
                    pulse.scanlineIdx = scanlineIndex                

                    #Find Min/Max for SPD file for zenith, azimuth, scanline and scanline index by checking each new pulse
                    if pulse.zenith < spdOutFile.zenithMin: spdOutFile.setZenithMin(pulse.zenith)
                    if pulse.zenith > spdOutFile.zenithMax: spdOutFile.setZenithMax(pulse.zenith)
                    if pulse.azimuth < spdOutFile.azimuthMin: spdOutFile.setAzimuthMin(pulse.azimuth)
                    if pulse.azimuth > spdOutFile.azimuthMax: spdOutFile.setAzimuthMax(pulse.azimuth)
                    if pulse.scanline < spdOutFile.scanlineMin: spdOutFile.setScanlineMin(pulse.scanline)
                    if pulse.scanline > spdOutFile.scanlineMax: spdOutFile.setScanlineMax(pulse.scanline)
                    if pulse.scanlineIdx < spdOutFile.scanlineIdxMin: spdOutFile.setScanlineIdxMin(pulse.scanlineIdx)
                    if pulse.scanlineIdx > spdOutFile.scanlineIdxMax: spdOutFile.setScanlineIdxMax(pulse.scanlineIdx)                


                    #Create point for first return
                    if numOfReturns != 0:
                        point = spdpy.createSPDPointPy()
                        point.returnID = 1
                        point.x = float(pulseDict["return1xyz"][0])
                        point.y = float(pulseDict["return1xyz"][1])
                        point.z = float(pulseDict["return1xyz"][2])
                        point.range = float(pulseDict["range1metres"])
                        point.amplitudeReturn = float(pulseDict["intensity1"])
                        pulse.pts.append(point)
                        pointsStartIdx += 1 
                        (xMin,xMax,yMin,yMax,zMin,zMax) = findSPDminMax(point.x,point.y,point.z,spdOutFile.xMin,spdOutFile.xMax,spdOutFile.yMin,spdOutFile.yMax,spdOutFile.zMin,spdOutFile.zMax)                   


                    #Create point for second return
                    if numOfReturns == 2:
                        point = spdpy.createSPDPointPy()
                        point.returnID = 2
                        point.x = float(pulseDict["return2xyz"][0])
                        point.y = float(pulseDict["return2xyz"][1])
                        point.z = float(pulseDict["return2xyz"][2])
                        point.range = float(pulseDict["range2metres"])
                        point.amplitudeReturn = float(pulseDict["intensity2"])
                        pulse.pts.append(point)
                        pointsStartIdx += 1
                        (xMin,xMax,yMin,yMax,zMin,zMax) = findSPDminMax(point.x,point.y,point.z,spdOutFile.xMin,spdOutFile.xMax,spdOutFile.yMin,spdOutFile.yMax,spdOutFile.zMin,spdOutFile.zMax)                   


                    #Set spd file min/max for x,y,z    
                    spdOutFile.setXMin(xMin)
                    spdOutFile.setXMax(xMax)
                    spdOutFile.setYMin(yMin)
                    spdOutFile.setYMax(yMax)
                    spdOutFile.setZMin(zMin)
                    spdOutFile.setZMax(zMax) 
                
                    
                #increment the pulse ID each time one is written out                
                pulseID += 1
                scanlineIndex += 1
                

            
            #add last pulse to scanline
            if hasSpd:
                if scanlineIndex == (maxScanLineID):
                    scanLineData[pulse.scanlineIdx].append(pulse)
            j+=1    
            
       
    # Write last scan line
    if hasSpd:
        spdWriter.writeDataRow(scanLineData, scanlineNumber-1)        
        # Close the output file                 
        spdWriter.close(spdOutFile)                    
                             
        #Convert the ouput spd file to a las file
        outFileLas = "%s.las" %(outputFileSpd.split('.')[0])
        cmd = "spdtranslate --if SPD --of LAS -i %s -o %s" %(outputFileSpd,outFileLas)
        print(cmd)
        if not os.path.exists(outFileLas):
            os.system(cmd)
        else:
            print("Las file %s already exists"%outFileLas)
    txtFile.close()
    

                
def main():
    
    cmdargs = getCmdargs()
    inFiles = cmdargs.inFiles
    cblVersion = cmdargs.cblVersion 
    agh = cmdargs.agh
    verbose = cmdargs.verbose
    if not hasSpd and verbose:
        print("No spdlib module found. No spd or las files will be produced.")
    print("Processing %s .gbl files" %len(cmdargs.inFiles))
    
    for inFile in cmdargs.inFiles:
        print("Currently processing %s" %inFile)
        doImport(inFile, cblVersion, agh, verbose)           
               
if __name__ == "__main__":    
# Use the class above to create the command args object
    main() 
