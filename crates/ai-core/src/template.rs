use std::collections::HashMap;
use tera::Tera;

pub struct TemplateEngine {
    tera: Tera,
}

impl TemplateEngine {
    pub fn new() -> Result<Self, tera::Error> {
        Ok(Self { tera: Tera::default() })
    }

    pub fn add_template(&mut self, name: &str, template: &str) -> Result<(), tera::Error> {
        self.tera.add_raw_template(name, template)
    }

    pub fn render(&self, name: &str, context: &HashMap<String, serde_json::Value>) -> Result<String, tera::Error> {
        let ctx = tera::Context::from_serialize(context)?;
        self.tera.render(name, &ctx)
    }
}
