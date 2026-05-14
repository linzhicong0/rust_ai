use clap::Parser;

#[derive(Parser)]
#[command(name = "ai")]
#[command(about = "AI Framework CLI — scaffold, run, test, deploy")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Scaffold a new project
    New { name: String },
    /// Run an agent or pipeline
    Run {
        /// Config file path
        #[arg(short, long)]
        config: Option<String>,
    },
    /// Serve the REST API
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::New { name } => {
            println!("Creating new project: {name}");
            // TODO: Implement project scaffolding
        }
        Commands::Run { config } => {
            println!("Running with config: {:?}", config);
            // TODO: Implement run command
        }
        Commands::Serve { port } => {
            println!("Starting server on port {port}");
            // TODO: Implement serve command
        }
    }

    Ok(())
}
