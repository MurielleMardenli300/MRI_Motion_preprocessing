import pydicom

ds = pydicom.dcmread("/volatile/Datasets/Varian_Motion/chum/10_28/mri/Mag_(0001)", stop_before_pixels=True, force=True)
for tag in ("MRAcquisitionType", "SliceThickness", "SpacingBetweenSlices", "NumberOfFrames",
            "Rows", "Columns", "PixelSpacing", "ScanningSequence", "SequenceVariant",
            "ScanOptions", "SeriesDescription", "ProtocolName"):
    print(f"{tag:22s} {ds.get(tag, '-')}")