[CmdletBinding()]
param(
    [switch]$ForceRedownload,
    [switch]$ForceReextract,
    [int64]$MinimumFreeBytes = 20GB
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.IO.Compression.FileSystem

# ============================================================
# Project paths
# ============================================================

$projectRoot = (
    Resolve-Path (Join-Path $PSScriptRoot "..")
).Path

$datasetRoot = Join-Path $projectRoot "data\raw\isic2019"
$archiveDir = Join-Path $datasetRoot "archives"
$metadataDir = Join-Path $datasetRoot "metadata"
$imageDir = Join-Path $datasetRoot "images"

$checksumDir = Join-Path $projectRoot "data\checksums\isic2019"
$checksumFile = Join-Path $checksumDir "isic2019_source_file_sha256.csv"

$auditDir = Join-Path $projectRoot "reports\dataset_audits"
$auditFile = Join-Path $auditDir "isic2019_acquisition_audit.json"
$completionMarker = Join-Path $datasetRoot "isic2019_acquisition_complete.json"

$trainingArchive = Join-Path `
    $archiveDir `
    "ISIC_2019_Training_Input.zip"

$groundTruthFile = Join-Path `
    $metadataDir `
    "ISIC_2019_Training_GroundTruth.csv"

$metadataFile = Join-Path `
    $metadataDir `
    "ISIC_2019_Training_Metadata.csv"

$expectedCount = 25331

$requiredDirectories = @(
    $archiveDir,
    $metadataDir,
    $checksumDir,
    $auditDir
)

foreach ($directory in $requiredDirectories) {
    New-Item `
        -ItemType Directory `
        -Path $directory `
        -Force |
        Out-Null
}

# A completion marker must never survive a failed or interrupted validation.
if (Test-Path $completionMarker) {
    Remove-Item $completionMarker -Force
}

# ============================================================
# Official source definitions
# ============================================================

$sourceFiles = @(
    [PSCustomObject]@{
        Key = "training_images"
        Name = "ISIC 2019 training images"
        Url = "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip"
        Destination = $trainingArchive
    },
    [PSCustomObject]@{
        Key = "ground_truth"
        Name = "ISIC 2019 training ground truth"
        Url = "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv"
        Destination = $groundTruthFile
    },
    [PSCustomObject]@{
        Key = "metadata"
        Name = "ISIC 2019 training metadata"
        Url = "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Metadata.csv"
        Destination = $metadataFile
    }
)

# ============================================================
# Helper functions
# ============================================================

function Get-RemoteContentLength {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url
    )

    try {
        $response = Invoke-WebRequest `
            -Uri $Url `
            -Method Head `
            -UseBasicParsing

        $contentLength = $response.Headers["Content-Length"]

        if ([string]::IsNullOrWhiteSpace($contentLength)) {
            return [int64]0
        }

        return [int64]$contentLength
    }
    catch {
        Write-Warning (
            "Could not read remote Content-Length for {0}. " +
            "Final integrity checks will still run." -f $Url
        )

        return [int64]0
    }
}

function Assert-FreeDiskSpace {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [int64]$RequiredBytes
    )

    $driveRoot = [System.IO.Path]::GetPathRoot($Path)
    $driveName = $driveRoot.TrimEnd("\").TrimEnd(":")
    $drive = Get-PSDrive -Name $driveName

    $freeGb = [math]::Round($drive.Free / 1GB, 2)
    $requiredGb = [math]::Round($RequiredBytes / 1GB, 2)

    Write-Host "[STORAGE] Free space: $freeGb GB"
    Write-Host "[STORAGE] Minimum required: $requiredGb GB"

    if ($drive.Free -lt $RequiredBytes) {
        throw (
            "Insufficient free disk space. " +
            "Free: $freeGb GB; required: $requiredGb GB."
        )
    }
}

function Download-SourceFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$Url,

        [Parameter(Mandatory = $true)]
        [string]$Destination,

        [Parameter(Mandatory = $true)]
        [bool]$Redownload
    )

    $remoteLength = Get-RemoteContentLength -Url $Url

    if ($Redownload -and (Test-Path $Destination)) {
        Remove-Item $Destination -Force
    }

    if (Test-Path $Destination) {
        $localLength = (Get-Item $Destination).Length

        if (
            ($remoteLength -gt 0 -and $localLength -eq $remoteLength) -or
            ($remoteLength -eq 0 -and $localLength -gt 0)
        ) {
            Write-Host "[SKIP] $Name already exists."
            Write-Host "       $Destination"
            return
        }

        Write-Warning (
            "Existing file size is inconsistent. " +
            "It will be downloaded again: $Destination"
        )

        Remove-Item $Destination -Force
    }

    $temporaryFile = "$Destination.partial"

    if (
        (Test-Path $temporaryFile) -and
        $remoteLength -gt 0 -and
        (Get-Item $temporaryFile).Length -gt $remoteLength
    ) {
        Remove-Item $temporaryFile -Force
    }

    Write-Host ""
    Write-Host "[DOWNLOAD] $Name"
    Write-Host "           $Url"

    $curlCommand = Get-Command "curl.exe" -ErrorAction SilentlyContinue

    if ($null -ne $curlCommand) {
        & curl.exe `
            --location `
            --fail `
            --retry 5 `
            --retry-delay 5 `
            --continue-at - `
            --output $temporaryFile `
            $Url

        if ($LASTEXITCODE -ne 0) {
            throw "curl.exe failed while downloading: $Name"
        }
    }
    else {
        if (Test-Path $temporaryFile) {
            Remove-Item $temporaryFile -Force
        }

        $bitsCommand = Get-Command `
            "Start-BitsTransfer" `
            -ErrorAction SilentlyContinue

        if ($null -ne $bitsCommand) {
            try {
                Start-BitsTransfer `
                    -Source $Url `
                    -Destination $temporaryFile `
                    -DisplayName $Name `
                    -Description "Downloading official ISIC 2019 source data"
            }
            catch {
                Write-Warning (
                    "BITS failed. Falling back to Invoke-WebRequest."
                )

                Invoke-WebRequest `
                    -Uri $Url `
                    -OutFile $temporaryFile `
                    -UseBasicParsing
            }
        }
        else {
            Invoke-WebRequest `
                -Uri $Url `
                -OutFile $temporaryFile `
                -UseBasicParsing
        }
    }

    if (
        -not (Test-Path $temporaryFile) -or
        (Get-Item $temporaryFile).Length -eq 0
    ) {
        throw "Download failed or produced an empty file: $Name"
    }

    $downloadedLength = (Get-Item $temporaryFile).Length

    if (
        $remoteLength -gt 0 -and
        $downloadedLength -ne $remoteLength
    ) {
        throw (
            "Downloaded size mismatch for $Name. " +
            "Expected $remoteLength bytes; found $downloadedLength bytes. " +
            "The partial file was retained for a resumable retry."
        )
    }

    Move-Item `
        -Path $temporaryFile `
        -Destination $Destination `
        -Force

    Write-Host "[DONE] $Destination"
}

function Assert-TrainingArchive {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [int]$ExpectedJpegCount
    )

    Write-Host ""
    Write-Host "[ZIP] Validating archive structure..."

    $archive = $null

    try {
        $archive = [System.IO.Compression.ZipFile]::OpenRead($Path)

        $jpegEntries = @(
            $archive.Entries |
                Where-Object {
                    $_.FullName -match "\.jpe?g$"
                }
        )

        if ($jpegEntries.Count -ne $ExpectedJpegCount) {
            throw (
                "ZIP image-entry validation failed. " +
                "Expected $ExpectedJpegCount JPEG entries; " +
                "found $($jpegEntries.Count)."
            )
        }
    }
    finally {
        if ($null -ne $archive) {
            $archive.Dispose()
        }
    }

    Write-Host "[PASS] ZIP contains $ExpectedJpegCount JPEG entries."
}

function Import-ValidatedCsv {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string[]]$RequiredColumns,

        [Parameter(Mandatory = $true)]
        [int]$ExpectedRows,

        [Parameter(Mandatory = $true)]
        [string]$DisplayName
    )

    $rows = @(Import-Csv -LiteralPath $Path)

    if ($rows.Count -ne $ExpectedRows) {
        throw (
            "$DisplayName row-count validation failed. " +
            "Expected $ExpectedRows; found $($rows.Count)."
        )
    }

    if ($rows.Count -eq 0) {
        throw "$DisplayName is empty."
    }

    $headers = @($rows[0].PSObject.Properties.Name)

    foreach ($requiredColumn in $RequiredColumns) {
        if ($headers -notcontains $requiredColumn) {
            throw (
                "$DisplayName is missing required column: " +
                $requiredColumn
            )
        }
    }

    return $rows
}

function Assert-UniqueValues {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Values,

        [Parameter(Mandatory = $true)]
        [string]$DisplayName
    )

    $duplicates = @(
        $Values |
            Group-Object |
            Where-Object {
                $_.Count -gt 1
            }
    )

    if ($duplicates.Count -gt 0) {
        $examples = (
            $duplicates |
                Select-Object -First 10 |
                ForEach-Object {
                    $_.Name
                }
        ) -join ", "

        throw (
            "$DisplayName contains duplicate identifiers. " +
            "Examples: $examples"
        )
    }
}

function Assert-OneHotGroundTruth {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Rows,

        [Parameter(Mandatory = $true)]
        [string[]]$DiagnosisColumns
    )

    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    $invalidRows = New-Object System.Collections.Generic.List[string]

    foreach ($row in $Rows) {
        $rowSum = [double]0

        foreach ($column in $DiagnosisColumns) {
            try {
                $value = [double]::Parse(
                    [string]$row.$column,
                    $culture
                )
            }
            catch {
                throw (
                    "Ground-truth value is not numeric. " +
                    "Image: $($row.image); column: $column"
                )
            }

            if ($value -ne 0.0 -and $value -ne 1.0) {
                throw (
                    "Ground-truth value is not binary. " +
                    "Image: $($row.image); column: $column; " +
                    "value: $value"
                )
            }

            $rowSum += $value
        }

        if ([math]::Abs($rowSum - 1.0) -gt 0.000001) {
            $invalidRows.Add([string]$row.image)
        }
    }

    if ($invalidRows.Count -gt 0) {
        $examples = (
            $invalidRows |
                Select-Object -First 10
        ) -join ", "

        throw (
            "Ground-truth one-hot validation failed for " +
            "$($invalidRows.Count) rows. Examples: $examples"
        )
    }
}

function Assert-IdentifierSetsMatch {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$ReferenceIds,

        [Parameter(Mandatory = $true)]
        [string[]]$ObservedIds,

        [Parameter(Mandatory = $true)]
        [string]$ObservedName
    )

    $comparison = @(
        Compare-Object `
            -ReferenceObject $ReferenceIds `
            -DifferenceObject $ObservedIds
    )

    $missing = @(
        $comparison |
            Where-Object {
                $_.SideIndicator -eq "<="
            }
    )

    $unexpected = @(
        $comparison |
            Where-Object {
                $_.SideIndicator -eq "=>"
            }
    )

    if ($missing.Count -gt 0 -or $unexpected.Count -gt 0) {
        throw (
            "$ObservedName identifier mismatch. " +
            "Missing: $($missing.Count); " +
            "unexpected: $($unexpected.Count)."
        )
    }
}

function Expand-TrainingImages {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ArchivePath,

        [Parameter(Mandatory = $true)]
        [string]$DestinationPath
    )

    if (Test-Path $DestinationPath) {
        Remove-Item `
            -Path $DestinationPath `
            -Recurse `
            -Force
    }

    New-Item `
        -ItemType Directory `
        -Path $DestinationPath `
        -Force |
        Out-Null

    Write-Host ""
    Write-Host "[EXTRACT] Extracting training images..."

    try {
        Expand-Archive `
            -LiteralPath $ArchivePath `
            -DestinationPath $DestinationPath `
            -Force
    }
    catch {
        if (Test-Path $DestinationPath) {
            Remove-Item `
                -Path $DestinationPath `
                -Recurse `
                -Force
        }

        throw
    }

    Write-Host "[DONE] Image extraction completed."
}

function Assert-DatasetConsistency {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$GroundTruthRows,

        [Parameter(Mandatory = $true)]
        [object[]]$MetadataRows,

        [Parameter(Mandatory = $true)]
        [string]$ImagesPath,

        [Parameter(Mandatory = $true)]
        [int]$ExpectedRows
    )

    $groundTruthIds = @(
        $GroundTruthRows |
            ForEach-Object {
                [string]$_.image
            } |
            Sort-Object
    )

    $metadataIds = @(
        $MetadataRows |
            ForEach-Object {
                [string]$_.image
            } |
            Sort-Object
    )

    Assert-UniqueValues `
        -Values $groundTruthIds `
        -DisplayName "Ground truth"

    Assert-UniqueValues `
        -Values $metadataIds `
        -DisplayName "Metadata"

    $imageFiles = @(
        Get-ChildItem `
            -LiteralPath $ImagesPath `
            -File `
            -Recurse |
            Where-Object {
                $_.Extension -match "^\.jpe?g$"
            }
    )

    if ($imageFiles.Count -ne $ExpectedRows) {
        throw (
            "Extracted image-count validation failed. " +
            "Expected $ExpectedRows; found $($imageFiles.Count)."
        )
    }

    $imageIds = @(
        $imageFiles |
            ForEach-Object {
                $_.BaseName
            } |
            Sort-Object
    )

    Assert-UniqueValues `
        -Values $imageIds `
        -DisplayName "Extracted images"

    Assert-IdentifierSetsMatch `
        -ReferenceIds $groundTruthIds `
        -ObservedIds $metadataIds `
        -ObservedName "Metadata"

    Assert-IdentifierSetsMatch `
        -ReferenceIds $groundTruthIds `
        -ObservedIds $imageIds `
        -ObservedName "Extracted images"

    return $imageFiles
}

# ============================================================
# Preflight and acquisition
# ============================================================

Assert-FreeDiskSpace `
    -Path $datasetRoot `
    -RequiredBytes $MinimumFreeBytes

foreach ($sourceFile in $sourceFiles) {
    Download-SourceFile `
        -Name $sourceFile.Name `
        -Url $sourceFile.Url `
        -Destination $sourceFile.Destination `
        -Redownload ([bool]$ForceRedownload)
}

Assert-TrainingArchive `
    -Path $trainingArchive `
    -ExpectedJpegCount $expectedCount

$diagnosisColumns = @(
    "MEL",
    "NV",
    "BCC",
    "AK",
    "BKL",
    "DF",
    "VASC",
    "SCC",
    "UNK"
)

$groundTruthRows = @(
    Import-ValidatedCsv `
        -Path $groundTruthFile `
        -RequiredColumns (@("image") + $diagnosisColumns) `
        -ExpectedRows $expectedCount `
        -DisplayName "ISIC 2019 ground truth"
)

$metadataRows = @(
    Import-ValidatedCsv `
        -Path $metadataFile `
        -RequiredColumns @(
            "image",
            "age_approx",
            "anatom_site_general",
            "lesion_id",
            "sex"
        ) `
        -ExpectedRows $expectedCount `
        -DisplayName "ISIC 2019 metadata"
)

Assert-OneHotGroundTruth `
    -Rows $groundTruthRows `
    -DiagnosisColumns $diagnosisColumns

# ============================================================
# Extraction with recovery from incomplete prior state
# ============================================================

$reuseExistingExtraction = $false

if (
    -not $ForceReextract -and
    (Test-Path $imageDir)
) {
    $candidateCount = @(
        Get-ChildItem `
            -LiteralPath $imageDir `
            -File `
            -Recurse |
            Where-Object {
                $_.Extension -match "^\.jpe?g$"
            }
    ).Count

    if ($candidateCount -eq $expectedCount) {
        $reuseExistingExtraction = $true
        Write-Host ""
        Write-Host (
            "[CHECK] Existing extraction has the expected " +
            "image count. Full consistency validation will follow."
        )
    }
}

if (-not $reuseExistingExtraction) {
    Expand-TrainingImages `
        -ArchivePath $trainingArchive `
        -DestinationPath $imageDir
}

try {
    $imageFiles = @(
        Assert-DatasetConsistency `
            -GroundTruthRows $groundTruthRows `
            -MetadataRows $metadataRows `
            -ImagesPath $imageDir `
            -ExpectedRows $expectedCount
    )
}
catch {
    if ($reuseExistingExtraction) {
        Write-Warning (
            "Existing extraction failed consistency validation. " +
            "A clean extraction will be performed."
        )

        Expand-TrainingImages `
            -ArchivePath $trainingArchive `
            -DestinationPath $imageDir

        $imageFiles = @(
            Assert-DatasetConsistency `
                -GroundTruthRows $groundTruthRows `
                -MetadataRows $metadataRows `
                -ImagesPath $imageDir `
                -ExpectedRows $expectedCount
        )
    }
    else {
        throw
    }
}

# ============================================================
# Source checksums
# ============================================================

Write-Host ""
Write-Host "[CHECKSUM] Calculating SHA-256 checksums..."

$verifiedAtUtc = (Get-Date).ToUniversalTime().ToString("o")

$checksumRecords = foreach ($sourceFile in $sourceFiles) {
    $hash = Get-FileHash `
        -Path $sourceFile.Destination `
        -Algorithm SHA256

    [PSCustomObject]@{
        dataset = "isic2019"
        source_key = $sourceFile.Key
        source_file = Split-Path `
            $sourceFile.Destination `
            -Leaf
        source_url = $sourceFile.Url
        sha256 = $hash.Hash.ToLowerInvariant()
        size_bytes = (
            Get-Item $sourceFile.Destination
        ).Length
        verified_at_utc = $verifiedAtUtc
    }
}

$checksumRecords |
    Export-Csv `
        -Path $checksumFile `
        -NoTypeInformation `
        -Encoding UTF8

# ============================================================
# Machine-readable acquisition audit
# ============================================================

$classDistribution = [ordered]@{}

foreach ($column in $diagnosisColumns) {
    $classDistribution[$column] = @(
        $groundTruthRows |
            Where-Object {
                [double]::Parse(
                    [string]$_.$column,
                    [System.Globalization.CultureInfo]::InvariantCulture
                ) -eq 1.0
            }
    ).Count
}

$metadataHeaders = @(
    $metadataRows[0].PSObject.Properties.Name
)

$nonEmptyLesionIdCount = @(
    $metadataRows |
        Where-Object {
            -not [string]::IsNullOrWhiteSpace(
                [string]$_.lesion_id
            )
        }
).Count

$audit = [ordered]@{
    dataset = "isic2019"
    acquisition_status = "complete"
    validated_at_utc = $verifiedAtUtc
    expected_record_count = $expectedCount
    counts = [ordered]@{
        images = $imageFiles.Count
        ground_truth_rows = $groundTruthRows.Count
        metadata_rows = $metadataRows.Count
    }
    validation = [ordered]@{
        zip_opened_successfully = $true
        zip_jpeg_entry_count_valid = $true
        csv_required_columns_valid = $true
        ground_truth_one_hot_valid = $true
        image_ids_unique = $true
        ground_truth_ids_unique = $true
        metadata_ids_unique = $true
        image_ground_truth_ids_match = $true
        metadata_ground_truth_ids_match = $true
    }
    identifier_availability = [ordered]@{
        patient_id_column_available = (
            $metadataHeaders -contains "patient_id"
        )
        lesion_id_column_available = (
            $metadataHeaders -contains "lesion_id"
        )
        non_empty_lesion_id_count = $nonEmptyLesionIdCount
    }
    class_distribution = $classDistribution
    source_checksum_file = (
        "data/checksums/isic2019/" +
        "isic2019_source_file_sha256.csv"
    )
}

$audit |
    ConvertTo-Json -Depth 8 |
    Set-Content `
        -Path $auditFile `
        -Encoding UTF8

# The marker is created only after every preceding operation succeeds.
$completionRecord = [ordered]@{
    dataset = "isic2019"
    status = "complete"
    completed_at_utc = $verifiedAtUtc
    image_count = $imageFiles.Count
    ground_truth_rows = $groundTruthRows.Count
    metadata_rows = $metadataRows.Count
    audit_file = (
        "reports/dataset_audits/" +
        "isic2019_acquisition_audit.json"
    )
    checksum_file = (
        "data/checksums/isic2019/" +
        "isic2019_source_file_sha256.csv"
    )
}

$completionRecord |
    ConvertTo-Json -Depth 5 |
    Set-Content `
        -Path $completionMarker `
        -Encoding UTF8

# ============================================================
# Final summary
# ============================================================

Write-Host ""
Write-Host "=================================================="
Write-Host "ISIC 2019 acquisition and integrity checks passed"
Write-Host "=================================================="
Write-Host "Dataset root:        $datasetRoot"
Write-Host "Images:              $($imageFiles.Count)"
Write-Host "Ground-truth rows:   $($groundTruthRows.Count)"
Write-Host "Metadata rows:       $($metadataRows.Count)"
Write-Host "Checksum file:       $checksumFile"
Write-Host "Acquisition audit:   $auditFile"
Write-Host "Completion marker:   $completionMarker"
