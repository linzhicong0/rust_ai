// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Video Understanding (REQ-8.4)
//!
//! Provides frame extraction, temporal context, scene boundary detection, and
//! a composable video analysis pipeline for video understanding workflows.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MS_PER_INPUT_BYTE: u64 = 100;
const SYNTHETIC_FRAME_SIZE: usize = 32;
const DEFAULT_FRAME_WIDTH: u32 = 8;
const DEFAULT_FRAME_HEIGHT: u32 = 4;

/// Raw frame data extracted from a video stream.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VideoFrame {
    /// Frame timestamp in milliseconds from the start of the video.
    pub timestamp_ms: u64,
    /// Encoded or raw frame bytes.
    pub data: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel or container format for the frame data.
    pub format: FrameFormat,
}

/// Supported frame formats for extracted frames.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FrameFormat {
    Rgb,
    Rgba,
    Jpeg,
    Png,
}

/// Configuration for extracting frames from a video source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FrameExtractionConfig {
    /// Interval between extracted frames in milliseconds.
    pub interval_ms: u64,
    /// Optional maximum number of frames to extract.
    pub max_frames: Option<usize>,
    /// Optional start offset in milliseconds.
    pub start_ms: Option<u64>,
    /// Optional end offset in milliseconds.
    pub end_ms: Option<u64>,
}

impl FrameExtractionConfig {
    /// Create a new extraction configuration with the required interval.
    pub fn new(interval_ms: u64) -> Self {
        Self {
            interval_ms,
            max_frames: None,
            start_ms: None,
            end_ms: None,
        }
    }

    /// Set the maximum number of extracted frames.
    pub fn with_max_frames(mut self, max_frames: usize) -> Self {
        self.max_frames = Some(max_frames);
        self
    }

    /// Set the starting timestamp.
    pub fn with_start_ms(mut self, start_ms: u64) -> Self {
        self.start_ms = Some(start_ms);
        self
    }

    /// Set the ending timestamp.
    pub fn with_end_ms(mut self, end_ms: u64) -> Self {
        self.end_ms = Some(end_ms);
        self
    }
}

/// Detected scene transition marker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SceneBoundary {
    /// Timestamp where the scene boundary occurs.
    pub timestamp_ms: u64,
    /// Confidence score in the range [0.0, 1.0].
    pub confidence: f32,
    /// Optional human-readable description of the boundary.
    pub description: Option<String>,
}

/// Multi-frame temporal context for downstream analysis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemporalContext {
    /// Extracted frames included in the temporal window.
    pub frames: Vec<VideoFrame>,
    /// Covered duration in milliseconds.
    pub duration_ms: u64,
    /// Scene boundaries detected across the frames.
    pub scene_boundaries: Vec<SceneBoundary>,
}

/// Description of a single scene spanning a time range.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SceneDescription {
    /// Scene start timestamp in milliseconds.
    pub start_ms: u64,
    /// Scene end timestamp in milliseconds.
    pub end_ms: u64,
    /// Human-readable scene description.
    pub description: String,
    /// Confidence score in the range [0.0, 1.0].
    pub confidence: f32,
}

/// Final result of a video analysis pass.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VideoAnalysisResult {
    /// Scene descriptions generated from boundary detection.
    pub scenes: Vec<SceneDescription>,
    /// Temporal context used during analysis.
    pub temporal_context: TemporalContext,
    /// Total number of frames analyzed.
    pub frame_count: usize,
}

/// Errors that can occur during video analysis.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum VideoError {
    /// Invalid extraction or analysis configuration.
    #[error("Invalid video config: {0}")]
    InvalidConfig(String),

    /// Frame extraction failed.
    #[error("Frame extraction failed: {0}")]
    ExtractionFailed(String),

    /// Analysis failed after extraction.
    #[error("Video analysis failed: {0}")]
    AnalysisFailed(String),

    /// The provided video contains no data.
    #[error("Video is empty")]
    EmptyVideo,

    /// Requested frame range falls outside the synthetic video duration.
    #[error("Requested frame range is outside the video bounds")]
    FrameOutOfBounds,
}

/// Extracts frames from video data.
#[async_trait]
pub trait FrameExtractor: Send + Sync {
    /// Extract frames using the supplied configuration.
    async fn extract_frames(
        &self,
        video_data: &[u8],
        config: &FrameExtractionConfig,
    ) -> Result<Vec<VideoFrame>, VideoError>;
}

/// Detects scene boundaries across a sequence of frames.
#[async_trait]
pub trait SceneDetector: Send + Sync {
    /// Detect scene boundaries from extracted frames.
    async fn detect_scenes(&self, frames: &[VideoFrame]) -> Result<Vec<SceneBoundary>, VideoError>;
}

/// End-to-end video analyzer with temporal understanding support.
#[async_trait]
pub trait VideoAnalyzer: Send + Sync {
    /// Analyze video data and return scene descriptions plus temporal context.
    async fn analyze(
        &self,
        video_data: &[u8],
        config: &FrameExtractionConfig,
    ) -> Result<VideoAnalysisResult, VideoError>;
}

/// In-memory frame extractor that simulates frame extraction from raw bytes.
#[derive(Debug, Clone)]
pub struct InMemoryFrameExtractor {
    /// Width assigned to synthetic frames.
    pub width: u32,
    /// Height assigned to synthetic frames.
    pub height: u32,
    /// Format assigned to synthetic frames.
    pub format: FrameFormat,
}

impl Default for InMemoryFrameExtractor {
    fn default() -> Self {
        Self {
            width: DEFAULT_FRAME_WIDTH,
            height: DEFAULT_FRAME_HEIGHT,
            format: FrameFormat::Rgb,
        }
    }
}

impl InMemoryFrameExtractor {
    /// Create a new synthetic extractor.
    pub fn new() -> Self {
        Self::default()
    }

    fn validate_config(
        &self,
        video_data: &[u8],
        config: &FrameExtractionConfig,
    ) -> Result<u64, VideoError> {
        if video_data.is_empty() {
            return Err(VideoError::EmptyVideo);
        }
        if config.interval_ms == 0 {
            return Err(VideoError::InvalidConfig(
                "frame extraction interval must be greater than zero".to_string(),
            ));
        }
        if matches!(config.max_frames, Some(0)) {
            return Err(VideoError::InvalidConfig(
                "max_frames must be greater than zero when provided".to_string(),
            ));
        }
        if let (Some(start_ms), Some(end_ms)) = (config.start_ms, config.end_ms) {
            if start_ms > end_ms {
                return Err(VideoError::InvalidConfig(
                    "start_ms cannot be greater than end_ms".to_string(),
                ));
            }
        }

        let duration_ms = synthetic_duration_ms(video_data)?;

        if config
            .start_ms
            .is_some_and(|start_ms| start_ms > duration_ms)
            || config.end_ms.is_some_and(|end_ms| end_ms > duration_ms)
        {
            return Err(VideoError::FrameOutOfBounds);
        }

        Ok(duration_ms)
    }
}

#[async_trait]
impl FrameExtractor for InMemoryFrameExtractor {
    async fn extract_frames(
        &self,
        video_data: &[u8],
        config: &FrameExtractionConfig,
    ) -> Result<Vec<VideoFrame>, VideoError> {
        let duration_ms = self.validate_config(video_data, config)?;
        let start_ms = config.start_ms.unwrap_or(0);
        let end_ms = config.end_ms.unwrap_or(duration_ms);
        let mut timestamps = Vec::new();
        let mut current_ms = start_ms;

        while current_ms <= end_ms {
            timestamps.push(current_ms);
            if let Some(max_frames) = config.max_frames {
                if timestamps.len() >= max_frames {
                    break;
                }
            }

            match current_ms.checked_add(config.interval_ms) {
                Some(next_ms) if next_ms > current_ms => current_ms = next_ms,
                _ => {
                    return Err(VideoError::ExtractionFailed(
                        "failed to advance frame timestamp".to_string(),
                    ))
                }
            }
        }

        Ok(timestamps
            .into_iter()
            .map(|timestamp_ms| VideoFrame {
                timestamp_ms,
                data: synthetic_frame_bytes(video_data, timestamp_ms, duration_ms),
                width: self.width,
                height: self.height,
                format: self.format.clone(),
            })
            .collect())
    }
}

/// Scene detector based on simple byte-wise frame differences.
#[derive(Debug, Clone)]
pub struct SimpleSceneDetector {
    /// Normalized threshold above which a frame transition is considered a new scene.
    pub difference_threshold: f32,
}

impl Default for SimpleSceneDetector {
    fn default() -> Self {
        Self {
            difference_threshold: 0.25,
        }
    }
}

impl SimpleSceneDetector {
    /// Create a detector with the given threshold.
    pub fn new(difference_threshold: f32) -> Self {
        Self {
            difference_threshold,
        }
    }
}

#[async_trait]
impl SceneDetector for SimpleSceneDetector {
    async fn detect_scenes(&self, frames: &[VideoFrame]) -> Result<Vec<SceneBoundary>, VideoError> {
        if frames.len() < 2 {
            return Ok(Vec::new());
        }

        let mut boundaries = Vec::new();
        for pair in frames.windows(2) {
            let previous = &pair[0];
            let current = &pair[1];
            let difference = normalized_frame_difference(previous, current)?;

            if difference >= self.difference_threshold {
                boundaries.push(SceneBoundary {
                    timestamp_ms: current.timestamp_ms,
                    confidence: difference,
                    description: Some("Scene change detected from visual difference".to_string()),
                });
            }
        }

        Ok(boundaries)
    }
}

/// Default end-to-end analyzer composed from a frame extractor and scene detector.
#[derive(Debug, Clone)]
pub struct DefaultVideoAnalyzer<E = InMemoryFrameExtractor, D = SimpleSceneDetector> {
    extractor: E,
    scene_detector: D,
}

impl<E, D> DefaultVideoAnalyzer<E, D> {
    /// Create a new analyzer from the provided components.
    pub fn new(extractor: E, scene_detector: D) -> Self {
        Self {
            extractor,
            scene_detector,
        }
    }
}

impl Default for DefaultVideoAnalyzer<InMemoryFrameExtractor, SimpleSceneDetector> {
    fn default() -> Self {
        Self::new(
            InMemoryFrameExtractor::default(),
            SimpleSceneDetector::default(),
        )
    }
}

#[async_trait]
impl<E, D> VideoAnalyzer for DefaultVideoAnalyzer<E, D>
where
    E: FrameExtractor,
    D: SceneDetector,
{
    async fn analyze(
        &self,
        video_data: &[u8],
        config: &FrameExtractionConfig,
    ) -> Result<VideoAnalysisResult, VideoError> {
        let frames = self.extractor.extract_frames(video_data, config).await?;
        if frames.is_empty() {
            return Err(VideoError::EmptyVideo);
        }

        let scene_boundaries = self.scene_detector.detect_scenes(&frames).await?;
        let duration_ms = frames
            .last()
            .map(|frame| frame.timestamp_ms)
            .unwrap_or_default()
            .saturating_sub(
                frames
                    .first()
                    .map(|frame| frame.timestamp_ms)
                    .unwrap_or_default(),
            );
        let temporal_context = TemporalContext {
            frames: frames.clone(),
            duration_ms,
            scene_boundaries: scene_boundaries.clone(),
        };
        let scenes = build_scene_descriptions(&frames, &scene_boundaries);

        Ok(VideoAnalysisResult {
            frame_count: frames.len(),
            scenes,
            temporal_context,
        })
    }
}

fn synthetic_duration_ms(video_data: &[u8]) -> Result<u64, VideoError> {
    if video_data.is_empty() {
        return Err(VideoError::EmptyVideo);
    }

    Ok((video_data.len() as u64).saturating_mul(MS_PER_INPUT_BYTE))
}

fn synthetic_frame_bytes(video_data: &[u8], timestamp_ms: u64, duration_ms: u64) -> Vec<u8> {
    let denominator = duration_ms.max(1) as usize;
    let base_index = ((timestamp_ms.min(duration_ms) as usize) * video_data.len()) / denominator;
    let base_index = base_index.min(video_data.len().saturating_sub(1));
    let segment_len = (video_data.len() / 4).max(1);
    let segment_end = (base_index + segment_len).min(video_data.len());
    let segment = &video_data[base_index..segment_end];

    (0..SYNTHETIC_FRAME_SIZE)
        .map(|offset| {
            let source_value = segment[offset % segment.len()];
            let temporal_offset =
                (((timestamp_ms / MS_PER_INPUT_BYTE) as u8) % 8) + (offset as u8 % 3);
            source_value.saturating_add(temporal_offset)
        })
        .collect()
}

fn normalized_frame_difference(
    previous: &VideoFrame,
    current: &VideoFrame,
) -> Result<f32, VideoError> {
    let compared = previous.data.len().min(current.data.len());
    if compared == 0 {
        return Err(VideoError::AnalysisFailed(
            "cannot compare empty frame payloads".to_string(),
        ));
    }

    let total_difference: u64 = previous
        .data
        .iter()
        .zip(current.data.iter())
        .map(|(left, right)| u64::from(u8::abs_diff(*left, *right)))
        .sum();
    let average_difference = total_difference as f32 / compared as f32;
    let length_penalty = previous.data.len().abs_diff(current.data.len()) as f32
        / previous.data.len().max(current.data.len()).max(1) as f32;

    Ok((average_difference / 255.0 + length_penalty).clamp(0.0, 1.0))
}

fn build_scene_descriptions(
    frames: &[VideoFrame],
    boundaries: &[SceneBoundary],
) -> Vec<SceneDescription> {
    let Some(first_frame) = frames.first() else {
        return Vec::new();
    };
    let end_of_video = frames
        .last()
        .map(|frame| frame.timestamp_ms)
        .unwrap_or_default();

    if boundaries.is_empty() {
        return vec![SceneDescription {
            start_ms: first_frame.timestamp_ms,
            end_ms: end_of_video,
            description: "Single continuous scene".to_string(),
            confidence: 1.0,
        }];
    }

    let mut scenes = Vec::new();
    let mut scene_start = first_frame.timestamp_ms;

    for boundary in boundaries {
        if boundary.timestamp_ms < scene_start {
            continue;
        }

        scenes.push(SceneDescription {
            start_ms: scene_start,
            end_ms: boundary.timestamp_ms,
            description: boundary.description.clone().unwrap_or_else(|| {
                format!("Scene from {scene_start}ms to {}ms", boundary.timestamp_ms)
            }),
            confidence: boundary.confidence,
        });
        scene_start = boundary.timestamp_ms;
    }

    scenes.push(SceneDescription {
        start_ms: scene_start,
        end_ms: end_of_video,
        description: format!("Scene from {scene_start}ms to {end_of_video}ms"),
        confidence: boundaries
            .last()
            .map(|boundary| boundary.confidence)
            .unwrap_or(1.0),
    });

    scenes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_video_data() -> Vec<u8> {
        vec![0u8; 10].into_iter().chain(vec![255u8; 10]).collect()
    }

    #[test]
    fn test_frame_extractor_is_object_safe() {
        let _: Option<Box<dyn FrameExtractor>> = None;
    }

    #[test]
    fn test_scene_detector_is_object_safe() {
        let _: Option<Box<dyn SceneDetector>> = None;
    }

    #[test]
    fn test_video_analyzer_is_object_safe() {
        let _: Option<Box<dyn VideoAnalyzer>> = None;
    }

    #[tokio::test]
    async fn test_frame_extraction_with_configurable_intervals() {
        let extractor = InMemoryFrameExtractor::default();
        let config = FrameExtractionConfig::new(500);

        let frames = extractor
            .extract_frames(&sample_video_data(), &config)
            .await
            .unwrap();

        let timestamps: Vec<u64> = frames.iter().map(|frame| frame.timestamp_ms).collect();
        assert_eq!(timestamps, vec![0, 500, 1000, 1500, 2000]);
        assert!(frames
            .iter()
            .all(|frame| frame.width == DEFAULT_FRAME_WIDTH));
        assert!(frames
            .iter()
            .all(|frame| frame.height == DEFAULT_FRAME_HEIGHT));
    }

    #[tokio::test]
    async fn test_frame_extraction_with_start_end_boundaries() {
        let extractor = InMemoryFrameExtractor::default();
        let config = FrameExtractionConfig::new(400)
            .with_start_ms(400)
            .with_end_ms(1200);

        let frames = extractor
            .extract_frames(&sample_video_data(), &config)
            .await
            .unwrap();

        let timestamps: Vec<u64> = frames.iter().map(|frame| frame.timestamp_ms).collect();
        assert_eq!(timestamps, vec![400, 800, 1200]);
    }

    #[tokio::test]
    async fn test_scene_boundary_detection() {
        let detector = SimpleSceneDetector::new(0.2);
        let frames = vec![
            VideoFrame {
                timestamp_ms: 0,
                data: vec![0; 8],
                width: 2,
                height: 2,
                format: FrameFormat::Rgb,
            },
            VideoFrame {
                timestamp_ms: 1000,
                data: vec![0; 8],
                width: 2,
                height: 2,
                format: FrameFormat::Rgb,
            },
            VideoFrame {
                timestamp_ms: 2000,
                data: vec![255; 8],
                width: 2,
                height: 2,
                format: FrameFormat::Rgb,
            },
        ];

        let boundaries = detector.detect_scenes(&frames).await.unwrap();

        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].timestamp_ms, 2000);
        assert!(boundaries[0].confidence >= 0.9);
    }

    #[tokio::test]
    async fn test_full_video_analysis_pipeline() {
        let analyzer = DefaultVideoAnalyzer::new(
            InMemoryFrameExtractor::default(),
            SimpleSceneDetector::new(0.05),
        );
        let config = FrameExtractionConfig::new(500);
        let result = analyzer
            .analyze(&sample_video_data(), &config)
            .await
            .unwrap();

        assert_eq!(result.frame_count, 5);
        assert_eq!(result.temporal_context.frames.len(), 5);
        assert_eq!(result.temporal_context.duration_ms, 2000);
        assert!(!result.temporal_context.scene_boundaries.is_empty());
        assert!(result.scenes.len() >= 2);
        assert_eq!(result.scenes.first().unwrap().start_ms, 0);
    }

    #[tokio::test]
    async fn test_empty_video_error() {
        let extractor = InMemoryFrameExtractor::default();
        let err = extractor
            .extract_frames(&[], &FrameExtractionConfig::new(100))
            .await
            .unwrap_err();

        assert_eq!(err, VideoError::EmptyVideo);
    }

    #[tokio::test]
    async fn test_invalid_config_error() {
        let extractor = InMemoryFrameExtractor::default();
        let err = extractor
            .extract_frames(&sample_video_data(), &FrameExtractionConfig::new(0))
            .await
            .unwrap_err();

        assert_eq!(
            err,
            VideoError::InvalidConfig(
                "frame extraction interval must be greater than zero".to_string()
            )
        );
    }

    #[tokio::test]
    async fn test_frame_out_of_bounds_error() {
        let extractor = InMemoryFrameExtractor::default();
        let err = extractor
            .extract_frames(
                &sample_video_data(),
                &FrameExtractionConfig::new(100).with_start_ms(2500),
            )
            .await
            .unwrap_err();

        assert_eq!(err, VideoError::FrameOutOfBounds);
    }
}
