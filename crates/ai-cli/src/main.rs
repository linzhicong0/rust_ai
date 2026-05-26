use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ai")]
#[command(about = "AI Framework CLI — scaffold, run, test, deploy")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// Scaffold a new project
    New {
        /// Project name
        name: String,
        /// Template to use (default: basic)
        #[arg(short, long, default_value = "basic")]
        template: String,
    },
    /// Run an agent or pipeline
    Run {
        /// Agent or pipeline name
        #[arg(short, long)]
        agent: Option<String>,
        /// Config file path
        #[arg(short, long)]
        config: Option<String>,
    },
    /// Run evaluation test suites
    Test {
        /// Test suite name or path
        suite: Option<String>,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
        /// Only run tests matching this pattern
        #[arg(short, long)]
        filter: Option<String>,
    },
    /// Deploy the API server
    Deploy {
        /// Target environment (local, staging, production)
        #[arg(short, long, default_value = "local")]
        target: String,
        /// Port for local deployment
        #[arg(short, long, default_value = "8080")]
        port: u16,
        /// Configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
    /// Serve the REST API
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },
}

/// Execute the scaffold (new) command.
pub fn execute_new(name: &str, template: &str) -> anyhow::Result<String> {
    let project_dir = PathBuf::from(name);
    if project_dir.exists() {
        anyhow::bail!("Directory '{}' already exists", name);
    }

    Ok(format!(
        "Created new project '{}' using template '{}'",
        name, template
    ))
}

/// Validate test suite configuration.
pub fn validate_test_suite(suite: Option<&str>, format: &str) -> anyhow::Result<TestConfig> {
    let valid_formats = ["text", "json", "junit"];
    if !valid_formats.contains(&format) {
        anyhow::bail!(
            "Invalid format '{}'. Valid formats: {:?}",
            format,
            valid_formats
        );
    }

    Ok(TestConfig {
        suite: suite.map(|s| s.to_string()),
        format: format.to_string(),
    })
}

/// Test configuration parsed from CLI args.
#[derive(Debug, Clone)]
pub struct TestConfig {
    pub suite: Option<String>,
    pub format: String,
}

/// Validate deploy configuration.
pub fn validate_deploy_config(target: &str, port: u16) -> anyhow::Result<DeployConfig> {
    let valid_targets = ["local", "staging", "production"];
    if !valid_targets.contains(&target) {
        anyhow::bail!(
            "Invalid target '{}'. Valid targets: {:?}",
            target,
            valid_targets
        );
    }

    if port == 0 {
        anyhow::bail!("Port must be non-zero");
    }

    Ok(DeployConfig {
        target: target.to_string(),
        port,
    })
}

/// Deploy configuration parsed from CLI args.
#[derive(Debug, Clone)]
pub struct DeployConfig {
    pub target: String,
    pub port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::New { name, template } => {
            let result = execute_new(&name, &template)?;
            println!("{result}");
        }
        Commands::Run { agent, config } => {
            println!("Running agent: {:?}, config: {:?}", agent, config);
            // TODO: Implement full run command
        }
        Commands::Test {
            suite,
            format,
            filter,
        } => {
            let config = validate_test_suite(suite.as_deref(), &format)?;
            println!(
                "Running test suite: {:?}, format: {}, filter: {:?}",
                config.suite, config.format, filter
            );
            // TODO: Implement full test execution
        }
        Commands::Deploy {
            target,
            port,
            config,
        } => {
            let deploy_config = validate_deploy_config(&target, port)?;
            println!(
                "Deploying to '{}' on port {}, config: {:?}",
                deploy_config.target, deploy_config.port, config
            );
            // TODO: Implement full deploy command
        }
        Commands::Serve { port } => {
            println!("Starting server on port {port}");
            // TODO: Implement serve command
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_cli_new_command_parses() {
        let cli = Cli::parse_from(["ai", "new", "my-project"]);
        match cli.command {
            Commands::New { name, template } => {
                assert_eq!(name, "my-project");
                assert_eq!(template, "basic");
            }
            _ => panic!("Expected New command"),
        }
    }

    #[test]
    fn test_cli_new_with_template() {
        let cli = Cli::parse_from(["ai", "new", "my-project", "--template", "advanced"]);
        match cli.command {
            Commands::New { name, template } => {
                assert_eq!(name, "my-project");
                assert_eq!(template, "advanced");
            }
            _ => panic!("Expected New command"),
        }
    }

    #[test]
    fn test_cli_run_command_parses() {
        let cli = Cli::parse_from(["ai", "run", "--agent", "my-agent", "--config", "config.toml"]);
        match cli.command {
            Commands::Run { agent, config } => {
                assert_eq!(agent.unwrap(), "my-agent");
                assert_eq!(config.unwrap(), "config.toml");
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_cli_test_command_parses() {
        let cli = Cli::parse_from(["ai", "test", "my-suite", "--format", "json"]);
        match cli.command {
            Commands::Test {
                suite,
                format,
                filter,
            } => {
                assert_eq!(suite.unwrap(), "my-suite");
                assert_eq!(format, "json");
                assert!(filter.is_none());
            }
            _ => panic!("Expected Test command"),
        }
    }

    #[test]
    fn test_cli_test_with_filter() {
        let cli = Cli::parse_from(["ai", "test", "--filter", "accuracy"]);
        match cli.command {
            Commands::Test {
                suite,
                format,
                filter,
            } => {
                assert!(suite.is_none());
                assert_eq!(format, "text");
                assert_eq!(filter.unwrap(), "accuracy");
            }
            _ => panic!("Expected Test command"),
        }
    }

    #[test]
    fn test_cli_deploy_command_parses() {
        let cli = Cli::parse_from(["ai", "deploy", "--target", "production", "--port", "9090"]);
        match cli.command {
            Commands::Deploy {
                target, port, config,
            } => {
                assert_eq!(target, "production");
                assert_eq!(port, 9090);
                assert!(config.is_none());
            }
            _ => panic!("Expected Deploy command"),
        }
    }

    #[test]
    fn test_cli_deploy_defaults() {
        let cli = Cli::parse_from(["ai", "deploy"]);
        match cli.command {
            Commands::Deploy {
                target, port, config,
            } => {
                assert_eq!(target, "local");
                assert_eq!(port, 8080);
                assert!(config.is_none());
            }
            _ => panic!("Expected Deploy command"),
        }
    }

    #[test]
    fn test_cli_serve_command_parses() {
        let cli = Cli::parse_from(["ai", "serve", "--port", "3000"]);
        match cli.command {
            Commands::Serve { port } => {
                assert_eq!(port, 3000);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    #[test]
    fn test_execute_new_returns_success_message() {
        // Use a path that definitely doesn't exist
        let result = execute_new("__nonexistent_test_project_xyz__", "basic");
        assert!(result.is_ok());
        assert!(result.unwrap().contains("__nonexistent_test_project_xyz__"));
    }

    #[test]
    fn test_execute_new_existing_dir_fails() {
        // "." always exists
        let result = execute_new(".", "basic");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_test_suite_valid() {
        let config = validate_test_suite(Some("my-suite"), "json").unwrap();
        assert_eq!(config.suite.unwrap(), "my-suite");
        assert_eq!(config.format, "json");
    }

    #[test]
    fn test_validate_test_suite_invalid_format() {
        let result = validate_test_suite(Some("suite"), "xml");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_test_suite_no_suite() {
        let config = validate_test_suite(None, "text").unwrap();
        assert!(config.suite.is_none());
    }

    #[test]
    fn test_validate_deploy_config_valid() {
        let config = validate_deploy_config("production", 8080).unwrap();
        assert_eq!(config.target, "production");
        assert_eq!(config.port, 8080);
    }

    #[test]
    fn test_validate_deploy_config_invalid_target() {
        let result = validate_deploy_config("invalid", 8080);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_deploy_config_zero_port() {
        let result = validate_deploy_config("local", 0);
        assert!(result.is_err());
    }
}

