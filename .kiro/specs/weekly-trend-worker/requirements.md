# Requirements Document

## Introduction

This feature involves refactoring the existing news scanner to use specific high-quality RSS feeds and implementing a weekly worker system that automatically discovers trending topics and updates the database. The system will combine RSS feed analysis with the existing trend spotting capabilities to provide a comprehensive automated trend discovery pipeline.

## Requirements

### Requirement 1

**User Story:** As a system administrator, I want the news scanner to use only high-quality, reliable RSS feeds from major tech companies and platforms, so that the trend discovery is based on authoritative sources.

#### Acceptance Criteria

1. WHEN the news scanner is initialized THEN it SHALL use only the following RSS feeds:
   - https://netflixtechblog.com/feed
   - https://feeds.feedburner.com/GDBcode
   - https://engineering.fb.com/feed/
   - https://aws.amazon.com/blogs/aws/feed/
   - https://stripe.com/blog/engineering/feed.xml
   - https://news.ycombinator.com/rss

2. WHEN the news scanner processes feeds THEN it SHALL remove all previous RSS sources from the configuration

3. WHEN a feed is unavailable or returns an error THEN the system SHALL log the error and continue processing other feeds

### Requirement 2

**User Story:** As a content manager, I want a weekly worker that automatically discovers trending topics and exports them to CSV files uploaded to S3, so that the system stays current with tech trends without database dependencies.

#### Acceptance Criteria

1. WHEN the weekly worker runs THEN it SHALL combine data from the refactored news scanner and existing trend spotter

2. WHEN trend analysis is complete THEN the worker SHALL export the results to CSV format with timestamps

3. WHEN the worker encounters errors THEN it SHALL log detailed error information and continue processing

4. WHEN the worker completes successfully THEN it SHALL upload the CSV file to S3 and update a local status file indicating the last successful run

### Requirement 3

**User Story:** As a developer, I want the weekly worker to be easily schedulable and monitorable, so that I can integrate it with cron jobs or task schedulers and track its performance.

#### Acceptance Criteria

1. WHEN the worker is executed THEN it SHALL provide a standalone script that can be run independently

2. WHEN the worker runs THEN it SHALL log detailed progress information including start time, processing steps, and completion status

3. WHEN the worker completes THEN it SHALL return appropriate exit codes (0 for success, non-zero for failure)

4. WHEN the worker is scheduled THEN it SHALL handle concurrent execution gracefully by checking for existing running instances

### Requirement 4

**User Story:** As a system operator, I want the worker to export trending topics to standardized CSV files and upload them to S3, so that trending topics are properly stored and accessible to other system components without database dependencies.

#### Acceptance Criteria

1. WHEN trends are discovered THEN the worker SHALL format them into standardized CSV structure with consistent columns

2. WHEN exporting trend data THEN the worker SHALL include source attribution, confidence scores, and discovery timestamps in CSV columns

3. WHEN S3 upload fails THEN the worker SHALL retry with exponential backoff and store local backup copies

4. WHEN trends are exported THEN they SHALL be accessible through S3 with consistent naming patterns and metadata

### Requirement 5

**User Story:** As a content creator, I want the system to maintain historical trend data in S3, so that I can analyze trending patterns over time and avoid duplicate content.

#### Acceptance Criteria

1. WHEN new trends are discovered THEN the system SHALL preserve previous CSV exports with dated filenames in S3

2. WHEN exporting trends THEN the system SHALL include metadata columns about the discovery method and confidence level

3. WHEN trends are accessed THEN they SHALL be retrievable from S3 with timestamps in filenames for chronological sorting

4. WHEN duplicate trends are detected THEN the system SHALL include deduplication flags in the CSV export metadata