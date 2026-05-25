// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Audio Processing (REQ-8.3)
//!
//! Provides traits and types for speech-to-text (transcription) and text-to-speech
//! (synthesis) with multiple voice options.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during audio processing operations.
#[derive(Debug, Error)]
pub enum AudioError {
    /// The provider returned an API error.
    #[error("API error: {0}")]
    Api(String),

    /// Invalid input (e.g., unsupported format, empty audio).
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Rate limit exceeded.
    #[error("Rate limit exceeded: {0}")]
    RateLimited(String),

    /// Network or transport error.
    #[error("Transport error: {0}")]
    Transport(String),

    /// Unsupported operation for this provider.
    #[error("Unsupported: {0}")]
    Unsupported(String),
}

/// Supported audio formats for input/output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AudioFormat {
    Mp3,
    Wav,
    Flac,
    Ogg,
    Aac,
    Opus,
    Pcm,
}

/// Language specification for transcription/synthesis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Language {
    /// BCP-47 language code (e.g., "en-US", "fr-FR", "ja-JP").
    pub code: String,
}

impl Language {
    /// Create a new language specification.
    pub fn new(code: impl Into<String>) -> Self {
        Self { code: code.into() }
    }

    /// English (US).
    pub fn en_us() -> Self {
        Self::new("en-US")
    }

    /// English (UK).
    pub fn en_gb() -> Self {
        Self::new("en-GB")
    }

    /// French.
    pub fn fr() -> Self {
        Self::new("fr-FR")
    }

    /// Japanese.
    pub fn ja() -> Self {
        Self::new("ja-JP")
    }
}

/// Configuration for transcription requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeConfig {
    /// Language hint for the transcription engine.
    pub language: Option<Language>,
    /// Whether to include word-level timestamps.
    pub timestamps: bool,
    /// Whether to include speaker diarization.
    pub diarize: bool,
    /// Prompt or context to help guide transcription.
    pub prompt: Option<String>,
    /// Expected audio format.
    pub format: Option<AudioFormat>,
}

impl Default for TranscribeConfig {
    fn default() -> Self {
        Self {
            language: None,
            timestamps: false,
            diarize: false,
            prompt: None,
            format: None,
        }
    }
}

impl TranscribeConfig {
    /// Create a new config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set language hint.
    pub fn with_language(mut self, language: Language) -> Self {
        self.language = Some(language);
        self
    }

    /// Enable timestamps.
    pub fn with_timestamps(mut self) -> Self {
        self.timestamps = true;
        self
    }

    /// Enable speaker diarization.
    pub fn with_diarization(mut self) -> Self {
        self.diarize = true;
        self
    }

    /// Set the prompt/context.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set the audio format.
    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = Some(format);
        self
    }
}

/// A word with timing information from transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimedWord {
    /// The transcribed word.
    pub word: String,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
}

/// A segment of transcribed audio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    /// The transcribed text for this segment.
    pub text: String,
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Speaker ID (if diarization was enabled).
    pub speaker: Option<String>,
    /// Word-level timestamps (if enabled).
    pub words: Vec<TimedWord>,
}

/// Result of a transcription operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptResult {
    /// The full transcribed text.
    pub text: String,
    /// Detected language.
    pub language: Option<Language>,
    /// Duration of the audio in seconds.
    pub duration: f64,
    /// Segments with timing (if timestamps requested).
    pub segments: Vec<TranscriptSegment>,
}

/// Voice selection for speech synthesis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Voice {
    /// Voice identifier (provider-specific).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Language this voice supports.
    pub language: Language,
    /// Gender of the voice.
    pub gender: VoiceGender,
}

/// Gender classification for voices.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VoiceGender {
    Male,
    Female,
    Neutral,
}

/// Configuration for speech synthesis requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizeConfig {
    /// Voice to use for synthesis.
    pub voice: Voice,
    /// Output audio format.
    pub format: AudioFormat,
    /// Speaking speed (0.5 to 2.0, where 1.0 is normal).
    pub speed: f64,
    /// Pitch adjustment (-1.0 to 1.0).
    pub pitch: f64,
}

impl SynthesizeConfig {
    /// Create a new config with the given voice.
    pub fn new(voice: Voice) -> Self {
        Self {
            voice,
            format: AudioFormat::Mp3,
            speed: 1.0,
            pitch: 0.0,
        }
    }

    /// Set the output format.
    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the speaking speed.
    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = speed.clamp(0.5, 2.0);
        self
    }

    /// Set the pitch adjustment.
    pub fn with_pitch(mut self, pitch: f64) -> Self {
        self.pitch = pitch.clamp(-1.0, 1.0);
        self
    }
}

/// Result of a speech synthesis operation.
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// The generated audio data.
    pub audio_data: Vec<u8>,
    /// Audio format of the result.
    pub format: AudioFormat,
    /// Duration of the generated audio in seconds.
    pub duration: f64,
    /// Character count of the synthesized text.
    pub char_count: usize,
}

/// Trait for speech-to-text transcription.
#[async_trait]
pub trait Transcriber: Send + Sync {
    /// Transcribe audio data to text.
    async fn transcribe(
        &self,
        audio: &[u8],
        config: &TranscribeConfig,
    ) -> Result<TranscriptResult, AudioError>;

    /// Return the provider name.
    fn name(&self) -> &str;
}

/// Trait for text-to-speech synthesis.
#[async_trait]
pub trait Synthesizer: Send + Sync {
    /// Synthesize text to audio.
    async fn synthesize(
        &self,
        text: &str,
        config: &SynthesizeConfig,
    ) -> Result<SynthesisResult, AudioError>;

    /// List available voices.
    async fn list_voices(&self) -> Result<Vec<Voice>, AudioError>;

    /// Return the provider name.
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-8.3: Test Transcriber and Synthesizer traits

    #[test]
    fn test_language_constructors() {
        let en = Language::en_us();
        assert_eq!(en.code, "en-US");

        let fr = Language::fr();
        assert_eq!(fr.code, "fr-FR");

        let ja = Language::ja();
        assert_eq!(ja.code, "ja-JP");

        let custom = Language::new("de-DE");
        assert_eq!(custom.code, "de-DE");
    }

    #[test]
    fn test_transcribe_config_default() {
        let config = TranscribeConfig::default();
        assert!(config.language.is_none());
        assert!(!config.timestamps);
        assert!(!config.diarize);
        assert!(config.prompt.is_none());
        assert!(config.format.is_none());
    }

    #[test]
    fn test_transcribe_config_builder() {
        let config = TranscribeConfig::new()
            .with_language(Language::en_us())
            .with_timestamps()
            .with_diarization()
            .with_prompt("Meeting recording")
            .with_format(AudioFormat::Wav);

        assert_eq!(config.language.unwrap().code, "en-US");
        assert!(config.timestamps);
        assert!(config.diarize);
        assert_eq!(config.prompt.unwrap(), "Meeting recording");
        assert_eq!(config.format.unwrap(), AudioFormat::Wav);
    }

    #[test]
    fn test_synthesize_config() {
        let voice = Voice {
            id: "alloy".to_string(),
            name: "Alloy".to_string(),
            language: Language::en_us(),
            gender: VoiceGender::Neutral,
        };

        let config = SynthesizeConfig::new(voice.clone())
            .with_format(AudioFormat::Opus)
            .with_speed(1.5)
            .with_pitch(0.2);

        assert_eq!(config.voice.id, "alloy");
        assert_eq!(config.format, AudioFormat::Opus);
        assert_eq!(config.speed, 1.5);
        assert_eq!(config.pitch, 0.2);
    }

    #[test]
    fn test_synthesize_config_clamps_speed() {
        let voice = Voice {
            id: "nova".to_string(),
            name: "Nova".to_string(),
            language: Language::en_us(),
            gender: VoiceGender::Female,
        };

        let config = SynthesizeConfig::new(voice).with_speed(5.0);
        assert_eq!(config.speed, 2.0);

        let voice2 = Voice {
            id: "echo".to_string(),
            name: "Echo".to_string(),
            language: Language::en_us(),
            gender: VoiceGender::Male,
        };
        let config2 = SynthesizeConfig::new(voice2).with_speed(0.1);
        assert_eq!(config2.speed, 0.5);
    }

    #[test]
    fn test_synthesize_config_clamps_pitch() {
        let voice = Voice {
            id: "onyx".to_string(),
            name: "Onyx".to_string(),
            language: Language::en_us(),
            gender: VoiceGender::Male,
        };

        let config = SynthesizeConfig::new(voice).with_pitch(3.0);
        assert_eq!(config.pitch, 1.0);
    }

    #[test]
    fn test_timed_word() {
        let word = TimedWord {
            word: "hello".to_string(),
            start: 0.0,
            end: 0.5,
            confidence: 0.98,
        };
        assert_eq!(word.word, "hello");
        assert!(word.confidence > 0.9);
    }

    #[test]
    fn test_transcript_segment() {
        let segment = TranscriptSegment {
            text: "Hello world".to_string(),
            start: 0.0,
            end: 1.5,
            speaker: Some("Speaker 1".to_string()),
            words: vec![
                TimedWord {
                    word: "Hello".to_string(),
                    start: 0.0,
                    end: 0.5,
                    confidence: 0.99,
                },
                TimedWord {
                    word: "world".to_string(),
                    start: 0.6,
                    end: 1.5,
                    confidence: 0.95,
                },
            ],
        };

        assert_eq!(segment.words.len(), 2);
        assert_eq!(segment.speaker.unwrap(), "Speaker 1");
    }

    #[test]
    fn test_transcript_result() {
        let result = TranscriptResult {
            text: "Hello world, this is a test.".to_string(),
            language: Some(Language::en_us()),
            duration: 3.5,
            segments: vec![TranscriptSegment {
                text: "Hello world, this is a test.".to_string(),
                start: 0.0,
                end: 3.5,
                speaker: None,
                words: vec![],
            }],
        };

        assert_eq!(result.duration, 3.5);
        assert_eq!(result.segments.len(), 1);
    }

    #[test]
    fn test_synthesis_result() {
        let result = SynthesisResult {
            audio_data: vec![0u8; 1024],
            format: AudioFormat::Mp3,
            duration: 2.5,
            char_count: 50,
        };

        assert_eq!(result.audio_data.len(), 1024);
        assert_eq!(result.format, AudioFormat::Mp3);
        assert_eq!(result.duration, 2.5);
        assert_eq!(result.char_count, 50);
    }

    #[test]
    fn test_audio_error_display() {
        let err = AudioError::Api("service unavailable".to_string());
        assert!(err.to_string().contains("service unavailable"));

        let err = AudioError::InvalidInput("empty audio".to_string());
        assert!(err.to_string().contains("empty audio"));

        let err = AudioError::RateLimited("too many requests".to_string());
        assert!(err.to_string().contains("too many requests"));

        let err = AudioError::Transport("timeout".to_string());
        assert!(err.to_string().contains("timeout"));

        let err = AudioError::Unsupported("diarization not available".to_string());
        assert!(err.to_string().contains("diarization not available"));
    }

    // Test that traits are object-safe
    #[test]
    fn test_transcriber_is_object_safe() {
        let _: Option<Box<dyn Transcriber>> = None;
    }

    #[test]
    fn test_synthesizer_is_object_safe() {
        let _: Option<Box<dyn Synthesizer>> = None;
    }

    // Mock implementations for testing
    struct MockTranscriber;

    #[async_trait]
    impl Transcriber for MockTranscriber {
        async fn transcribe(
            &self,
            audio: &[u8],
            config: &TranscribeConfig,
        ) -> Result<TranscriptResult, AudioError> {
            if audio.is_empty() {
                return Err(AudioError::InvalidInput("empty audio data".to_string()));
            }

            let segments = if config.timestamps {
                vec![TranscriptSegment {
                    text: "Hello world".to_string(),
                    start: 0.0,
                    end: 1.5,
                    speaker: if config.diarize {
                        Some("Speaker 1".to_string())
                    } else {
                        None
                    },
                    words: vec![
                        TimedWord {
                            word: "Hello".to_string(),
                            start: 0.0,
                            end: 0.5,
                            confidence: 0.99,
                        },
                        TimedWord {
                            word: "world".to_string(),
                            start: 0.6,
                            end: 1.5,
                            confidence: 0.95,
                        },
                    ],
                }]
            } else {
                vec![]
            };

            Ok(TranscriptResult {
                text: "Hello world".to_string(),
                language: config.language.clone(),
                duration: 1.5,
                segments,
            })
        }

        fn name(&self) -> &str {
            "mock-whisper"
        }
    }

    struct MockSynthesizer;

    #[async_trait]
    impl Synthesizer for MockSynthesizer {
        async fn synthesize(
            &self,
            text: &str,
            config: &SynthesizeConfig,
        ) -> Result<SynthesisResult, AudioError> {
            if text.is_empty() {
                return Err(AudioError::InvalidInput("empty text".to_string()));
            }

            // Simulate audio generation - duration proportional to text length
            let duration = text.len() as f64 * 0.05 / config.speed;
            let audio_data = vec![0u8; (duration * 16000.0) as usize]; // Simulated PCM data

            Ok(SynthesisResult {
                audio_data,
                format: config.format.clone(),
                duration,
                char_count: text.len(),
            })
        }

        async fn list_voices(&self) -> Result<Vec<Voice>, AudioError> {
            Ok(vec![
                Voice {
                    id: "alloy".to_string(),
                    name: "Alloy".to_string(),
                    language: Language::en_us(),
                    gender: VoiceGender::Neutral,
                },
                Voice {
                    id: "echo".to_string(),
                    name: "Echo".to_string(),
                    language: Language::en_us(),
                    gender: VoiceGender::Male,
                },
                Voice {
                    id: "nova".to_string(),
                    name: "Nova".to_string(),
                    language: Language::en_us(),
                    gender: VoiceGender::Female,
                },
            ])
        }

        fn name(&self) -> &str {
            "mock-elevenlabs"
        }
    }

    #[tokio::test]
    async fn test_mock_transcriber_basic() {
        let transcriber = MockTranscriber;
        let audio = vec![1u8; 100];
        let config = TranscribeConfig::new();

        let result = transcriber.transcribe(&audio, &config).await.unwrap();
        assert_eq!(result.text, "Hello world");
        assert_eq!(result.duration, 1.5);
        assert!(result.segments.is_empty()); // No timestamps requested
    }

    #[tokio::test]
    async fn test_mock_transcriber_with_timestamps() {
        let transcriber = MockTranscriber;
        let audio = vec![1u8; 100];
        let config = TranscribeConfig::new()
            .with_language(Language::en_us())
            .with_timestamps();

        let result = transcriber.transcribe(&audio, &config).await.unwrap();
        assert_eq!(result.text, "Hello world");
        assert_eq!(result.language.unwrap().code, "en-US");
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0].words.len(), 2);
    }

    #[tokio::test]
    async fn test_mock_transcriber_with_diarization() {
        let transcriber = MockTranscriber;
        let audio = vec![1u8; 100];
        let config = TranscribeConfig::new().with_timestamps().with_diarization();

        let result = transcriber.transcribe(&audio, &config).await.unwrap();
        assert_eq!(result.segments[0].speaker, Some("Speaker 1".to_string()));
    }

    #[tokio::test]
    async fn test_mock_transcriber_empty_audio() {
        let transcriber = MockTranscriber;
        let audio: Vec<u8> = vec![];
        let config = TranscribeConfig::new();

        let result = transcriber.transcribe(&audio, &config).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            AudioError::InvalidInput(msg) => assert_eq!(msg, "empty audio data"),
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_mock_synthesizer_basic() {
        let synth = MockSynthesizer;
        let voice = Voice {
            id: "alloy".to_string(),
            name: "Alloy".to_string(),
            language: Language::en_us(),
            gender: VoiceGender::Neutral,
        };
        let config = SynthesizeConfig::new(voice);

        let result = synth.synthesize("Hello world", &config).await.unwrap();
        assert_eq!(result.char_count, 11);
        assert!(result.duration > 0.0);
        assert!(!result.audio_data.is_empty());
        assert_eq!(result.format, AudioFormat::Mp3);
    }

    #[tokio::test]
    async fn test_mock_synthesizer_empty_text() {
        let synth = MockSynthesizer;
        let voice = Voice {
            id: "alloy".to_string(),
            name: "Alloy".to_string(),
            language: Language::en_us(),
            gender: VoiceGender::Neutral,
        };
        let config = SynthesizeConfig::new(voice);

        let result = synth.synthesize("", &config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_synthesizer_list_voices() {
        let synth = MockSynthesizer;
        let voices = synth.list_voices().await.unwrap();
        assert_eq!(voices.len(), 3);
        assert_eq!(voices[0].id, "alloy");
        assert_eq!(voices[1].gender, VoiceGender::Male);
        assert_eq!(voices[2].gender, VoiceGender::Female);
    }

    #[test]
    fn test_transcriber_name() {
        let t = MockTranscriber;
        assert_eq!(t.name(), "mock-whisper");
    }

    #[test]
    fn test_synthesizer_name() {
        let s = MockSynthesizer;
        assert_eq!(s.name(), "mock-elevenlabs");
    }

    #[test]
    fn test_voice_equality() {
        let v1 = Voice {
            id: "alloy".to_string(),
            name: "Alloy".to_string(),
            language: Language::en_us(),
            gender: VoiceGender::Neutral,
        };
        let v2 = v1.clone();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_audio_format_variants() {
        let formats = vec![
            AudioFormat::Mp3,
            AudioFormat::Wav,
            AudioFormat::Flac,
            AudioFormat::Ogg,
            AudioFormat::Aac,
            AudioFormat::Opus,
            AudioFormat::Pcm,
        ];
        assert_eq!(formats.len(), 7);
    }
}
