// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Image Generation (REQ-8.2)
//!
//! Provides traits and types for image generation and editing through integration
//! with models like DALL-E, Stable Diffusion, and Midjourney.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during image generation operations.
#[derive(Debug, Error)]
pub enum ImageGenError {
    /// The provider returned an API error.
    #[error("API error: {0}")]
    Api(String),

    /// Invalid input parameters.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Rate limit exceeded.
    #[error("Rate limit exceeded: {0}")]
    RateLimited(String),

    /// Network or transport error.
    #[error("Transport error: {0}")]
    Transport(String),

    /// Unsupported operation for this provider.
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

/// Size specification for generated images.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImageSize {
    /// 256x256 pixels
    Small,
    /// 512x512 pixels
    Medium,
    /// 1024x1024 pixels
    Large,
    /// 1024x1792 pixels (portrait)
    Portrait,
    /// 1792x1024 pixels (landscape)
    Landscape,
    /// Custom dimensions
    Custom { width: u32, height: u32 },
}

impl ImageSize {
    /// Get the width and height as a tuple.
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Small => (256, 256),
            Self::Medium => (512, 512),
            Self::Large => (1024, 1024),
            Self::Portrait => (1024, 1792),
            Self::Landscape => (1792, 1024),
            Self::Custom { width, height } => (*width, *height),
        }
    }
}

/// Image output format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Jpeg,
    WebP,
}

/// Quality setting for image generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImageQuality {
    Standard,
    High,
    Ultra,
}

/// Style preset for image generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ImageStyle {
    Natural,
    Vivid,
    Artistic,
}

/// Configuration for image generation requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenConfig {
    /// Desired image size.
    pub size: ImageSize,
    /// Output format.
    pub format: ImageFormat,
    /// Quality setting.
    pub quality: ImageQuality,
    /// Style preset.
    pub style: ImageStyle,
    /// Number of images to generate.
    pub n: u32,
}

impl Default for ImageGenConfig {
    fn default() -> Self {
        Self {
            size: ImageSize::Large,
            format: ImageFormat::Png,
            quality: ImageQuality::Standard,
            style: ImageStyle::Natural,
            n: 1,
        }
    }
}

impl ImageGenConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the image size.
    pub fn with_size(mut self, size: ImageSize) -> Self {
        self.size = size;
        self
    }

    /// Set the output format.
    pub fn with_format(mut self, format: ImageFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the quality level.
    pub fn with_quality(mut self, quality: ImageQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Set the style.
    pub fn with_style(mut self, style: ImageStyle) -> Self {
        self.style = style;
        self
    }

    /// Set number of images to generate.
    pub fn with_n(mut self, n: u32) -> Self {
        self.n = n;
        self
    }
}

/// A generated image result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImage {
    /// The image data as bytes (base64-decoded) or URL.
    pub data: ImageData,
    /// Revised prompt (if the model revised the original prompt).
    pub revised_prompt: Option<String>,
    /// Image dimensions.
    pub width: u32,
    /// Image height.
    pub height: u32,
}

/// How image data is delivered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageData {
    /// Base64-encoded image bytes.
    Base64(String),
    /// URL to the generated image.
    Url(String),
}

/// Configuration for image editing requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEditConfig {
    /// The prompt describing the desired edit.
    pub prompt: String,
    /// Optional mask indicating the area to edit (base64 PNG).
    pub mask: Option<String>,
    /// Desired output size.
    pub size: ImageSize,
    /// Number of variations to generate.
    pub n: u32,
}

impl ImageEditConfig {
    /// Create a new edit config with a prompt.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            mask: None,
            size: ImageSize::Large,
            n: 1,
        }
    }

    /// Set a mask for the edit region.
    pub fn with_mask(mut self, mask: impl Into<String>) -> Self {
        self.mask = Some(mask.into());
        self
    }

    /// Set the output size.
    pub fn with_size(mut self, size: ImageSize) -> Self {
        self.size = size;
        self
    }

    /// Set number of variations.
    pub fn with_n(mut self, n: u32) -> Self {
        self.n = n;
        self
    }
}

/// Trait for image generation providers.
///
/// Implementations should support at least `generate()`. The `edit()` and `vary()`
/// methods may return `Unsupported` if the provider doesn't support them.
#[async_trait]
pub trait ImageGenerator: Send + Sync {
    /// Generate images from a text prompt.
    async fn generate(
        &self,
        prompt: &str,
        config: &ImageGenConfig,
    ) -> Result<Vec<GeneratedImage>, ImageGenError>;

    /// Edit an existing image based on a prompt and optional mask.
    async fn edit(
        &self,
        image: &[u8],
        config: &ImageEditConfig,
    ) -> Result<Vec<GeneratedImage>, ImageGenError>;

    /// Generate variations of an existing image.
    async fn vary(
        &self,
        image: &[u8],
        n: u32,
        size: &ImageSize,
    ) -> Result<Vec<GeneratedImage>, ImageGenError>;

    /// Return the provider name.
    fn name(&self) -> &str;
}

/// An image message type for agent consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMessage {
    /// Description or prompt used to generate the image.
    pub description: String,
    /// The image data.
    pub image: GeneratedImage,
    /// Provider that generated the image.
    pub provider: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-8.2: Test ImageGenerator trait definition and types

    #[test]
    fn test_image_size_dimensions() {
        assert_eq!(ImageSize::Small.dimensions(), (256, 256));
        assert_eq!(ImageSize::Medium.dimensions(), (512, 512));
        assert_eq!(ImageSize::Large.dimensions(), (1024, 1024));
        assert_eq!(ImageSize::Portrait.dimensions(), (1024, 1792));
        assert_eq!(ImageSize::Landscape.dimensions(), (1792, 1024));
        assert_eq!(
            ImageSize::Custom {
                width: 800,
                height: 600
            }
            .dimensions(),
            (800, 600)
        );
    }

    #[test]
    fn test_image_gen_config_default() {
        let config = ImageGenConfig::default();
        assert_eq!(config.size, ImageSize::Large);
        assert_eq!(config.format, ImageFormat::Png);
        assert_eq!(config.quality, ImageQuality::Standard);
        assert_eq!(config.style, ImageStyle::Natural);
        assert_eq!(config.n, 1);
    }

    #[test]
    fn test_image_gen_config_builder() {
        let config = ImageGenConfig::new()
            .with_size(ImageSize::Portrait)
            .with_format(ImageFormat::WebP)
            .with_quality(ImageQuality::High)
            .with_style(ImageStyle::Vivid)
            .with_n(3);

        assert_eq!(config.size, ImageSize::Portrait);
        assert_eq!(config.format, ImageFormat::WebP);
        assert_eq!(config.quality, ImageQuality::High);
        assert_eq!(config.style, ImageStyle::Vivid);
        assert_eq!(config.n, 3);
    }

    #[test]
    fn test_image_edit_config() {
        let config = ImageEditConfig::new("Add a sunset background")
            .with_mask("base64data")
            .with_size(ImageSize::Medium)
            .with_n(2);

        assert_eq!(config.prompt, "Add a sunset background");
        assert_eq!(config.mask, Some("base64data".to_string()));
        assert_eq!(config.size, ImageSize::Medium);
        assert_eq!(config.n, 2);
    }

    #[test]
    fn test_generated_image_creation() {
        let image = GeneratedImage {
            data: ImageData::Url("https://example.com/image.png".to_string()),
            revised_prompt: Some("A beautiful sunset over mountains".to_string()),
            width: 1024,
            height: 1024,
        };

        assert_eq!(image.width, 1024);
        assert_eq!(image.height, 1024);
        assert!(image.revised_prompt.is_some());
    }

    #[test]
    fn test_image_data_variants() {
        let url_data = ImageData::Url("https://example.com/img.png".to_string());
        let base64_data = ImageData::Base64("iVBORw0KGgo=".to_string());

        match url_data {
            ImageData::Url(u) => assert!(u.starts_with("https://")),
            _ => panic!("Expected Url variant"),
        }

        match base64_data {
            ImageData::Base64(b) => assert!(!b.is_empty()),
            _ => panic!("Expected Base64 variant"),
        }
    }

    #[test]
    fn test_image_message() {
        let msg = ImageMessage {
            description: "A cat sitting on a windowsill".to_string(),
            image: GeneratedImage {
                data: ImageData::Url("https://example.com/cat.png".to_string()),
                revised_prompt: None,
                width: 512,
                height: 512,
            },
            provider: "dall-e-3".to_string(),
        };

        assert_eq!(msg.provider, "dall-e-3");
        assert_eq!(msg.description, "A cat sitting on a windowsill");
        assert_eq!(msg.image.width, 512);
    }

    #[test]
    fn test_image_gen_error_display() {
        let err = ImageGenError::Api("quota exceeded".to_string());
        assert!(err.to_string().contains("quota exceeded"));

        let err = ImageGenError::InvalidInput("prompt too long".to_string());
        assert!(err.to_string().contains("prompt too long"));

        let err = ImageGenError::RateLimited("try again in 60s".to_string());
        assert!(err.to_string().contains("try again in 60s"));

        let err = ImageGenError::Transport("connection refused".to_string());
        assert!(err.to_string().contains("connection refused"));

        let err = ImageGenError::Unsupported("vary not supported".to_string());
        assert!(err.to_string().contains("vary not supported"));
    }

    // Test that the trait is object-safe
    #[test]
    fn test_image_generator_is_object_safe() {
        let _: Option<Box<dyn ImageGenerator>> = None;
    }

    #[test]
    fn test_image_gen_config_serialization() {
        let config = ImageGenConfig::new()
            .with_size(ImageSize::Large)
            .with_format(ImageFormat::Jpeg)
            .with_quality(ImageQuality::High)
            .with_n(2);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ImageGenConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.size, ImageSize::Large);
        assert_eq!(deserialized.format, ImageFormat::Jpeg);
        assert_eq!(deserialized.quality, ImageQuality::High);
        assert_eq!(deserialized.n, 2);
    }

    // Mock implementation for testing the trait
    struct MockImageGenerator;

    #[async_trait]
    impl ImageGenerator for MockImageGenerator {
        async fn generate(
            &self,
            prompt: &str,
            config: &ImageGenConfig,
        ) -> Result<Vec<GeneratedImage>, ImageGenError> {
            if prompt.is_empty() {
                return Err(ImageGenError::InvalidInput("empty prompt".to_string()));
            }
            let (width, height) = config.size.dimensions();
            let images = (0..config.n)
                .map(|i| GeneratedImage {
                    data: ImageData::Url(format!("https://example.com/gen_{}.png", i)),
                    revised_prompt: Some(format!("Revised: {}", prompt)),
                    width,
                    height,
                })
                .collect();
            Ok(images)
        }

        async fn edit(
            &self,
            _image: &[u8],
            config: &ImageEditConfig,
        ) -> Result<Vec<GeneratedImage>, ImageGenError> {
            let (width, height) = config.size.dimensions();
            let images = (0..config.n)
                .map(|i| GeneratedImage {
                    data: ImageData::Url(format!("https://example.com/edit_{}.png", i)),
                    revised_prompt: None,
                    width,
                    height,
                })
                .collect();
            Ok(images)
        }

        async fn vary(
            &self,
            _image: &[u8],
            n: u32,
            size: &ImageSize,
        ) -> Result<Vec<GeneratedImage>, ImageGenError> {
            let (width, height) = size.dimensions();
            let images = (0..n)
                .map(|i| GeneratedImage {
                    data: ImageData::Url(format!("https://example.com/vary_{}.png", i)),
                    revised_prompt: None,
                    width,
                    height,
                })
                .collect();
            Ok(images)
        }

        fn name(&self) -> &str {
            "mock-dall-e"
        }
    }

    #[tokio::test]
    async fn test_mock_image_generator_generate() {
        let gen = MockImageGenerator;
        let config = ImageGenConfig::new().with_n(2).with_size(ImageSize::Medium);

        let result = gen.generate("A beautiful sunset", &config).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].width, 512);
        assert_eq!(result[0].height, 512);
        assert!(result[0].revised_prompt.is_some());
    }

    #[tokio::test]
    async fn test_mock_image_generator_generate_empty_prompt() {
        let gen = MockImageGenerator;
        let config = ImageGenConfig::new();

        let result = gen.generate("", &config).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            ImageGenError::InvalidInput(msg) => assert_eq!(msg, "empty prompt"),
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_mock_image_generator_edit() {
        let gen = MockImageGenerator;
        let image_data = vec![0u8; 100];
        let config = ImageEditConfig::new("Make it brighter").with_n(1);

        let result = gen.edit(&image_data, &config).await.unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_mock_image_generator_vary() {
        let gen = MockImageGenerator;
        let image_data = vec![0u8; 100];

        let result = gen.vary(&image_data, 3, &ImageSize::Small).await.unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].width, 256);
        assert_eq!(result[0].height, 256);
    }

    #[test]
    fn test_image_generator_name() {
        let gen = MockImageGenerator;
        assert_eq!(gen.name(), "mock-dall-e");
    }
}
