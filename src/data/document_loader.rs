use docx_rs::*;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Loads a .docx file and extracts all plain text from it
pub fn load_docx(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let path = Path::new(path);

    if !path.exists() {
        return Err(format!("File not found: {}", path.display()).into());
    }

    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let docx = read_docx(&buf)?;

    let mut text = String::new();

    // Iterate over all document children and extract text
    for child in docx.document.children {
        match child {
            DocumentChild::Paragraph(paragraph) => {
                for para_child in paragraph.children {
                    match para_child {
                        ParagraphChild::Run(run) => {
                            for run_child in run.children {
                                match run_child {
                                    RunChild::Text(t) => {
                                        text.push_str(&t.text);
                                        text.push(' ');
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
                // Add newline after each paragraph
                text.push('\n');
            }
            DocumentChild::Table(table) => {
                // Extract text from tables too (useful for calendars)
                for row in table.rows {
                    match row {
                        TableChild::TableRow(tr) => {
                            for cell in tr.cells {
                                match cell {
                                    TableRowChild::TableCell(tc) => {
                                        for content in &tc.children {
                                            match content {
                                                TableCellContent::Paragraph(p) => {
                                                    for pc in &p.children {
                                                        match pc {
                                                            ParagraphChild::Run(run) => {
                                                                for rc in &run.children {
                                                                    match rc {
                                                                        RunChild::Text(t) => {
                                                                            text.push_str(&t.text);
                                                                            text.push(' ');
                                                                        }
                                                                        _ => {}
                                                                    }
                                                                }
                                                            }
                                                            _ => {}
                                                        }
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                        text.push('\t'); // separate cells with tab
                                    }
                                }
                            }
                            text.push('\n'); // new line per row
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(text.trim().to_string())
}

/// Loads multiple .docx files and combines their text
pub fn load_multiple_docx(paths: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    let mut combined = String::new();

    for path in paths {
        match load_docx(path) {
            Ok(text) => {
                combined.push_str(&text);
                combined.push_str("\n\n");
                println!("Loaded: {}", path);
            }
            Err(e) => {
                eprintln!("Warning: Could not load {}: {}", path, e);
            }
        }
    }

    if combined.is_empty() {
        return Err("No documents could be loaded".into());
    }

    Ok(combined)
}
/// Extracts only meaningful event lines from calendar text
/// Filters out day headers, date numbers, and grid structure
pub fn extract_events(text: &str) -> String {
    let day_names = ["SUNDAY","MONDAY","TUESDAY","WEDNESDAY",
        "THURSDAY","FRIDAY","SATURDAY"];

    let mut events = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        // Skip lines with 2+ day names (calendar grid headers)
        let day_count = day_names.iter()
            .filter(|d| trimmed.contains(*d))
            .count();
        if day_count >= 2 { continue; }

        // Count meaningful words (longer than 2 chars, not pure numbers)
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        let meaningful = words.iter()
            .filter(|w| w.len() > 2 && !w.chars().all(|c| c.is_numeric()))
            .count();

        if meaningful >= 2 {
            events.push(trimmed.to_string());
        }
    }

    events.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_nonexistent_file() {
        let result = load_docx("nonexistent.docx");
        assert!(result.is_err());
    }
}