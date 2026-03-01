//! Parser for the PRAM IR textual DSL format.
//!
//! Supports the following syntax:
//! ```text
//! algorithm NAME(param: type, ...) model MODEL {
//!     shared MEM: elem_type[size];
//!     processors = expr;
//!     // statements ...
//! }
//! ```

use super::ast::*;
use super::types::PramType;
use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A parse error with source location.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

impl ParseError {
    pub fn new(msg: impl Into<String>, line: usize, col: usize) -> Self {
        Self {
            message: msg.into(),
            line,
            column: col,
        }
    }

    /// Add context information to this parse error.
    pub fn with_context(mut self, ctx: &str) -> ParseError {
        self.message = format!("{} (in {})", self.message, ctx);
        self
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Parse error at {}:{}: {}",
            self.line, self.column, self.message
        )
    }
}

impl std::error::Error for ParseError {}

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TokenKind {
    // Keywords
    Algorithm,
    Model,
    Shared,
    Processors,
    ParallelFor,
    For,
    While,
    If,
    Else,
    Let,
    Barrier,
    SharedReadKw,
    SharedWriteKw,
    Return,
    In,
    Step,
    Pid,
    NumProcessors,
    AtomicCasKw,
    FetchAddKw,
    PrefixSumKw,
    AssertKw,
    NopKw,
    // Literals
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    Ident(String),
    StringLit(String),
    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    BangEq,
    AmpAmp,
    PipePipe,
    Amp,
    Pipe,
    Caret,
    Shl,
    Shr,
    Bang,
    Tilde,
    Eq,
    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Colon,
    DotDot,
    // End
    Eof,
}

#[derive(Debug, Clone)]
pub(crate) struct Token {
    kind: TokenKind,
    line: usize,
    col: usize,
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

struct Lexer {
    chars: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<char> {
        self.chars.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.chars.get(self.pos).copied()?;
        self.pos += 1;
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }

    fn skip_ws_comments(&mut self) {
        loop {
            while self.peek().map_or(false, |c| c.is_whitespace()) {
                self.advance();
            }
            if self.peek() == Some('/') && self.peek2() == Some('/') {
                while self.peek().map_or(false, |c| c != '\n') {
                    self.advance();
                }
                continue;
            }
            break;
        }
    }

    fn read_ident(&mut self) -> String {
        let mut s = String::new();
        while self.peek().map_or(false, |c| c.is_alphanumeric() || c == '_') {
            s.push(self.advance().unwrap());
        }
        s
    }

    fn read_number(&mut self) -> TokenKind {
        let mut s = String::new();
        let mut is_float = false;
        while self.peek().map_or(false, |c| c.is_ascii_digit()) {
            s.push(self.advance().unwrap());
        }
        // Check for float: '.' followed by digit (not '..')
        if self.peek() == Some('.') && self.peek2().map_or(false, |c| c.is_ascii_digit()) {
            is_float = true;
            s.push(self.advance().unwrap()); // '.'
            while self.peek().map_or(false, |c| c.is_ascii_digit()) {
                s.push(self.advance().unwrap());
            }
        }
        if is_float {
            TokenKind::FloatLit(s.parse().unwrap_or(0.0))
        } else {
            TokenKind::IntLit(s.parse().unwrap_or(0))
        }
    }

    fn read_string(&mut self) -> Result<String, ParseError> {
        let line = self.line;
        let col = self.col;
        self.advance(); // opening '"'
        let mut s = String::new();
        loop {
            match self.advance() {
                Some('"') => return Ok(s),
                Some('\\') => match self.advance() {
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('\\') => s.push('\\'),
                    Some('"') => s.push('"'),
                    Some(c) => s.push(c),
                    None => return Err(ParseError::new("Unterminated escape", line, col)),
                },
                Some(c) => s.push(c),
                None => return Err(ParseError::new("Unterminated string", line, col)),
            }
        }
    }

    fn classify(word: &str) -> TokenKind {
        match word {
            "algorithm" => TokenKind::Algorithm,
            "model" => TokenKind::Model,
            "shared" => TokenKind::Shared,
            "processors" => TokenKind::Processors,
            "parallel_for" => TokenKind::ParallelFor,
            "for" => TokenKind::For,
            "while" => TokenKind::While,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "let" => TokenKind::Let,
            "barrier" => TokenKind::Barrier,
            "shared_read" => TokenKind::SharedReadKw,
            "shared_write" => TokenKind::SharedWriteKw,
            "return" => TokenKind::Return,
            "in" => TokenKind::In,
            "step" => TokenKind::Step,
            "pid" => TokenKind::Pid,
            "num_processors" => TokenKind::NumProcessors,
            "atomic_cas" => TokenKind::AtomicCasKw,
            "fetch_add" => TokenKind::FetchAddKw,
            "prefix_sum" => TokenKind::PrefixSumKw,
            "assert" => TokenKind::AssertKw,
            "nop" => TokenKind::NopKw,
            "true" => TokenKind::BoolLit(true),
            "false" => TokenKind::BoolLit(false),
            _ => TokenKind::Ident(word.to_string()),
        }
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, ParseError> {
        let mut tokens = Vec::new();
        loop {
            self.skip_ws_comments();
            let line = self.line;
            let col = self.col;
            let ch = match self.peek() {
                Some(c) => c,
                None => {
                    tokens.push(Token { kind: TokenKind::Eof, line, col });
                    break;
                }
            };
            let kind = match ch {
                '+' => { self.advance(); TokenKind::Plus }
                '-' => { self.advance(); TokenKind::Minus }
                '*' => { self.advance(); TokenKind::Star }
                '/' => { self.advance(); TokenKind::Slash }
                '%' => { self.advance(); TokenKind::Percent }
                '~' => { self.advance(); TokenKind::Tilde }
                '^' => { self.advance(); TokenKind::Caret }
                '(' => { self.advance(); TokenKind::LParen }
                ')' => { self.advance(); TokenKind::RParen }
                '{' => { self.advance(); TokenKind::LBrace }
                '}' => { self.advance(); TokenKind::RBrace }
                '[' => { self.advance(); TokenKind::LBracket }
                ']' => { self.advance(); TokenKind::RBracket }
                ',' => { self.advance(); TokenKind::Comma }
                ';' => { self.advance(); TokenKind::Semi }
                ':' => { self.advance(); TokenKind::Colon }
                '<' => {
                    self.advance();
                    if self.peek() == Some('=') { self.advance(); TokenKind::Le }
                    else if self.peek() == Some('<') { self.advance(); TokenKind::Shl }
                    else { TokenKind::Lt }
                }
                '>' => {
                    self.advance();
                    if self.peek() == Some('=') { self.advance(); TokenKind::Ge }
                    else if self.peek() == Some('>') { self.advance(); TokenKind::Shr }
                    else { TokenKind::Gt }
                }
                '=' => {
                    self.advance();
                    if self.peek() == Some('=') { self.advance(); TokenKind::EqEq }
                    else { TokenKind::Eq }
                }
                '!' => {
                    self.advance();
                    if self.peek() == Some('=') { self.advance(); TokenKind::BangEq }
                    else { TokenKind::Bang }
                }
                '&' => {
                    self.advance();
                    if self.peek() == Some('&') { self.advance(); TokenKind::AmpAmp }
                    else { TokenKind::Amp }
                }
                '|' => {
                    self.advance();
                    if self.peek() == Some('|') { self.advance(); TokenKind::PipePipe }
                    else { TokenKind::Pipe }
                }
                '.' => {
                    self.advance();
                    if self.peek() == Some('.') {
                        self.advance();
                        TokenKind::DotDot
                    } else {
                        return Err(ParseError::new("Expected '..'", line, col));
                    }
                }
                '"' => {
                    let s = self.read_string()?;
                    TokenKind::StringLit(s)
                }
                c if c.is_ascii_digit() => self.read_number(),
                c if c.is_alphabetic() || c == '_' => {
                    let word = self.read_ident();
                    Self::classify(&word)
                }
                c => {
                    return Err(ParseError::new(
                        format!("Unexpected character: '{}'", c),
                        line,
                        col,
                    ));
                }
            };
            tokens.push(Token { kind, line, col });
        }
        Ok(tokens)
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &TokenKind {
        &self.tokens[self.pos.min(self.tokens.len() - 1)].kind
    }

    fn cur_loc(&self) -> (usize, usize) {
        let t = &self.tokens[self.pos.min(self.tokens.len() - 1)];
        (t.line, t.col)
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn err(&self, msg: impl Into<String>) -> ParseError {
        let (l, c) = self.cur_loc();
        ParseError::new(msg, l, c)
    }

    fn expect(&mut self, expected: TokenKind) -> Result<(), ParseError> {
        if *self.peek() == expected {
            self.advance();
            Ok(())
        } else {
            Err(self.err(format!("Expected {:?}, got {:?}", expected, self.peek())))
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(s) => {
                self.advance();
                Ok(s)
            }
            ref other => Err(self.err(format!("Expected identifier, got {:?}", other))),
        }
    }

    fn expect_string(&mut self) -> Result<String, ParseError> {
        match self.peek().clone() {
            TokenKind::StringLit(s) => {
                self.advance();
                Ok(s)
            }
            ref other => Err(self.err(format!("Expected string, got {:?}", other))),
        }
    }

    fn check(&self, kind: &TokenKind) -> bool {
        self.peek() == kind
    }

    // -- top-level ---------------------------------------------------------

    fn parse_program(&mut self) -> Result<PramProgram, ParseError> {
        self.expect(TokenKind::Algorithm)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::LParen)?;
        let params = self.parse_param_list()?;
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::Model)?;
        let model = self.parse_model()?;
        self.expect(TokenKind::LBrace)?;

        let mut shared_memory = Vec::new();
        let mut num_processors = Expr::IntLiteral(1);

        // Declarations at the top of the body.
        loop {
            if self.check(&TokenKind::Shared) {
                shared_memory.push(self.parse_shared_decl()?);
            } else if self.check(&TokenKind::Processors) {
                self.advance();
                self.expect(TokenKind::Eq)?;
                num_processors = self.parse_expr()?;
                self.expect(TokenKind::Semi)?;
            } else {
                break;
            }
        }

        let mut body = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
            body.push(self.parse_stmt()?);
        }
        self.expect(TokenKind::RBrace)?;

        Ok(PramProgram {
            name,
            memory_model: model,
            parameters: params,
            shared_memory,
            body,
            num_processors,
            work_bound: None,
            time_bound: None,
            description: None,
        })
    }

    fn parse_param_list(&mut self) -> Result<Vec<Parameter>, ParseError> {
        let mut params = Vec::new();
        if self.check(&TokenKind::RParen) {
            return Ok(params);
        }
        loop {
            let name = self.expect_ident()?;
            self.expect(TokenKind::Colon)?;
            let ty = self.parse_type()?;
            params.push(Parameter {
                name,
                param_type: ty,
            });
            if self.check(&TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        Ok(params)
    }

    fn parse_model(&mut self) -> Result<MemoryModel, ParseError> {
        let name = self.expect_ident()?;
        match name.as_str() {
            "EREW" => Ok(MemoryModel::EREW),
            "CREW" => Ok(MemoryModel::CREW),
            "CRCW_Priority" | "CRCWPriority" => Ok(MemoryModel::CRCWPriority),
            "CRCW_Arbitrary" | "CRCWArbitrary" => Ok(MemoryModel::CRCWArbitrary),
            "CRCW_Common" | "CRCWCommon" => Ok(MemoryModel::CRCWCommon),
            _ => Err(self.err(format!("Unknown memory model: {}", name))),
        }
    }

    fn parse_type(&mut self) -> Result<PramType, ParseError> {
        if self.check(&TokenKind::Shared) {
            self.advance();
            self.expect(TokenKind::Lt)?;
            let inner = self.parse_type()?;
            self.expect(TokenKind::Gt)?;
            return Ok(PramType::SharedMemory(Box::new(inner)));
        }
        let name = self.expect_ident()?;
        match name.as_str() {
            "i64" => Ok(PramType::Int64),
            "i32" => Ok(PramType::Int32),
            "f64" => Ok(PramType::Float64),
            "f32" => Ok(PramType::Float32),
            "bool" => Ok(PramType::Bool),
            _ => Err(self.err(format!("Unknown type: {}", name))),
        }
    }

    fn parse_shared_decl(&mut self) -> Result<SharedMemoryDecl, ParseError> {
        self.expect(TokenKind::Shared)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::Colon)?;
        let raw_ty = self.parse_type()?;
        // Unwrap SharedMemory wrapper if present.
        let elem_type = match raw_ty {
            PramType::SharedMemory(inner) => *inner,
            other => other,
        };
        self.expect(TokenKind::LBracket)?;
        let size = self.parse_expr()?;
        self.expect(TokenKind::RBracket)?;
        self.expect(TokenKind::Semi)?;
        Ok(SharedMemoryDecl {
            name,
            elem_type,
            size,
        })
    }

    // -- statements --------------------------------------------------------

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        match self.peek().clone() {
            TokenKind::ParallelFor => self.parse_parallel_for(),
            TokenKind::For => self.parse_seq_for(),
            TokenKind::While => self.parse_while(),
            TokenKind::If => self.parse_if(),
            TokenKind::Let => self.parse_let(),
            TokenKind::Barrier => {
                self.advance();
                self.expect(TokenKind::Semi)?;
                Ok(Stmt::Barrier)
            }
            TokenKind::NopKw => {
                self.advance();
                self.expect(TokenKind::Semi)?;
                Ok(Stmt::Nop)
            }
            TokenKind::SharedWriteKw => self.parse_shared_write_stmt(),
            TokenKind::Return => self.parse_return(),
            TokenKind::AtomicCasKw => self.parse_atomic_cas_stmt(),
            TokenKind::FetchAddKw => self.parse_fetch_add_stmt(),
            TokenKind::PrefixSumKw => self.parse_prefix_sum_stmt(),
            TokenKind::AssertKw => self.parse_assert_stmt(),
            TokenKind::LBrace => {
                self.advance();
                let mut stmts = Vec::new();
                while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
                    stmts.push(self.parse_stmt()?);
                }
                self.expect(TokenKind::RBrace)?;
                Ok(Stmt::Block(stmts))
            }
            TokenKind::Ident(_) => {
                // Look-ahead for `ident = atomic_cas(...)` or `ident = fetch_add(...)`
                // or plain assignment `ident = expr;`
                let is_assign = self.pos + 1 < self.tokens.len()
                    && self.tokens[self.pos + 1].kind == TokenKind::Eq;
                if is_assign {
                    let name = self.expect_ident()?;
                    self.expect(TokenKind::Eq)?;
                    // Check for atomic_cas / fetch_add after =
                    if self.check(&TokenKind::AtomicCasKw) {
                        return self.parse_atomic_cas_rhs(&name);
                    }
                    if self.check(&TokenKind::FetchAddKw) {
                        return self.parse_fetch_add_rhs(&name);
                    }
                    let expr = self.parse_expr()?;
                    self.expect(TokenKind::Semi)?;
                    Ok(Stmt::Assign(name, expr))
                } else {
                    let expr = self.parse_expr()?;
                    self.expect(TokenKind::Semi)?;
                    Ok(Stmt::ExprStmt(expr))
                }
            }
            _ => Err(self.err(format!("Expected statement, got {:?}", self.peek()))),
        }
    }

    fn parse_parallel_for(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::ParallelFor)?;
        let proc_var = self.expect_ident()?;
        self.expect(TokenKind::In)?;
        let _start = self.parse_expr()?; // typically 0 – discarded
        self.expect(TokenKind::DotDot)?;
        let num_procs = self.parse_expr()?;
        self.expect(TokenKind::LBrace)?;
        let mut body = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
            body.push(self.parse_stmt()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(Stmt::ParallelFor {
            proc_var,
            num_procs,
            body,
        })
    }

    fn parse_seq_for(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::For)?;
        let var = self.expect_ident()?;
        self.expect(TokenKind::In)?;
        let start = self.parse_expr()?;
        self.expect(TokenKind::DotDot)?;
        let end = self.parse_expr()?;
        let step = if self.check(&TokenKind::Step) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };
        self.expect(TokenKind::LBrace)?;
        let mut body = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
            body.push(self.parse_stmt()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(Stmt::SeqFor {
            var,
            start,
            end,
            step,
            body,
        })
    }

    fn parse_while(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::While)?;
        let condition = self.parse_expr()?;
        self.expect(TokenKind::LBrace)?;
        let mut body = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
            body.push(self.parse_stmt()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(Stmt::While { condition, body })
    }

    fn parse_if(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::If)?;
        let condition = self.parse_expr()?;
        self.expect(TokenKind::LBrace)?;
        let mut then_body = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
            then_body.push(self.parse_stmt()?);
        }
        self.expect(TokenKind::RBrace)?;
        let else_body = if self.check(&TokenKind::Else) {
            self.advance();
            if self.check(&TokenKind::If) {
                // else if
                vec![self.parse_if()?]
            } else {
                self.expect(TokenKind::LBrace)?;
                let mut eb = Vec::new();
                while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
                    eb.push(self.parse_stmt()?);
                }
                self.expect(TokenKind::RBrace)?;
                eb
            }
        } else {
            Vec::new()
        };
        Ok(Stmt::If {
            condition,
            then_body,
            else_body,
        })
    }

    fn parse_let(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::Let)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::Colon)?;
        let ty = self.parse_type()?;
        let init = if self.check(&TokenKind::Eq) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };
        self.expect(TokenKind::Semi)?;
        Ok(Stmt::LocalDecl(name, ty, init))
    }

    fn parse_shared_write_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::SharedWriteKw)?;
        self.expect(TokenKind::LParen)?;
        let memory = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let index = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let value = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::Semi)?;
        Ok(Stmt::SharedWrite {
            memory,
            index,
            value,
        })
    }

    fn parse_return(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::Return)?;
        if self.check(&TokenKind::Semi) {
            self.advance();
            Ok(Stmt::Return(None))
        } else {
            let expr = self.parse_expr()?;
            self.expect(TokenKind::Semi)?;
            Ok(Stmt::Return(Some(expr)))
        }
    }

    /// Parse: `atomic_cas(memory, index, expected, desired);`  (standalone, no result)
    fn parse_atomic_cas_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::AtomicCasKw)?;
        self.expect(TokenKind::LParen)?;
        let memory = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let index = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let expected = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let desired = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::Semi)?;
        Ok(Stmt::AtomicCAS {
            memory,
            index,
            expected,
            desired,
            result_var: "_".to_string(),
        })
    }

    /// Parse: `result = atomic_cas(memory, index, expected, desired);`
    fn parse_atomic_cas_rhs(&mut self, result_var: &str) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::AtomicCasKw)?;
        self.expect(TokenKind::LParen)?;
        let memory = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let index = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let expected = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let desired = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::Semi)?;
        Ok(Stmt::AtomicCAS {
            memory,
            index,
            expected,
            desired,
            result_var: result_var.to_string(),
        })
    }

    /// Parse: `fetch_add(memory, index, value);`  (standalone, no result)
    fn parse_fetch_add_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::FetchAddKw)?;
        self.expect(TokenKind::LParen)?;
        let memory = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let index = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let value = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::Semi)?;
        Ok(Stmt::FetchAdd {
            memory,
            index,
            value,
            result_var: "_".to_string(),
        })
    }

    /// Parse: `result = fetch_add(memory, index, value);`
    fn parse_fetch_add_rhs(&mut self, result_var: &str) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::FetchAddKw)?;
        self.expect(TokenKind::LParen)?;
        let memory = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let index = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let value = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::Semi)?;
        Ok(Stmt::FetchAdd {
            memory,
            index,
            value,
            result_var: result_var.to_string(),
        })
    }

    /// Parse: `prefix_sum(input, output, size, op);`
    fn parse_prefix_sum_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::PrefixSumKw)?;
        self.expect(TokenKind::LParen)?;
        let input = self.expect_ident()?;
        self.expect(TokenKind::Comma)?;
        let output = self.expect_ident()?;
        self.expect(TokenKind::Comma)?;
        let size = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let op = self.parse_binop_name()?;
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::Semi)?;
        Ok(Stmt::PrefixSum { input, output, size, op })
    }

    /// Parse a binary operator name like `+`, `-`, `*`, `min`, `max`.
    fn parse_binop_name(&mut self) -> Result<BinOp, ParseError> {
        match self.peek().clone() {
            TokenKind::Plus => { self.advance(); Ok(BinOp::Add) }
            TokenKind::Minus => { self.advance(); Ok(BinOp::Sub) }
            TokenKind::Star => { self.advance(); Ok(BinOp::Mul) }
            TokenKind::Ident(ref name) => {
                let op = match name.as_str() {
                    "min" => BinOp::Min,
                    "max" => BinOp::Max,
                    other => return Err(self.err(format!("Unknown operator: {}", other))),
                };
                self.advance();
                Ok(op)
            }
            _ => Err(self.err(format!("Expected operator, got {:?}", self.peek()))),
        }
    }

    /// Parse: `assert(expr, "message");`
    fn parse_assert_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.expect(TokenKind::AssertKw)?;
        self.expect(TokenKind::LParen)?;
        let expr = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let msg = self.expect_string()?;
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::Semi)?;
        Ok(Stmt::Assert(expr, msg))
    }

    // -- expressions (precedence climbing) ---------------------------------

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_and()?;
        while self.check(&TokenKind::PipePipe) {
            self.advance();
            let right = self.parse_and()?;
            left = Expr::binop(BinOp::Or, left, right);
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_bitor()?;
        while self.check(&TokenKind::AmpAmp) {
            self.advance();
            let right = self.parse_bitor()?;
            left = Expr::binop(BinOp::And, left, right);
        }
        Ok(left)
    }

    fn parse_bitor(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_bitxor()?;
        while self.check(&TokenKind::Pipe) {
            self.advance();
            let right = self.parse_bitxor()?;
            left = Expr::binop(BinOp::BitOr, left, right);
        }
        Ok(left)
    }

    fn parse_bitxor(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_bitand()?;
        while self.check(&TokenKind::Caret) {
            self.advance();
            let right = self.parse_bitand()?;
            left = Expr::binop(BinOp::BitXor, left, right);
        }
        Ok(left)
    }

    fn parse_bitand(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_equality()?;
        while self.check(&TokenKind::Amp) {
            self.advance();
            let right = self.parse_equality()?;
            left = Expr::binop(BinOp::BitAnd, left, right);
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_comparison()?;
        loop {
            let op = match self.peek() {
                TokenKind::EqEq => BinOp::Eq,
                TokenKind::BangEq => BinOp::Ne,
                _ => break,
            };
            self.advance();
            let right = self.parse_comparison()?;
            left = Expr::binop(op, left, right);
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_shift()?;
        loop {
            let op = match self.peek() {
                TokenKind::Lt => BinOp::Lt,
                TokenKind::Le => BinOp::Le,
                TokenKind::Gt => BinOp::Gt,
                TokenKind::Ge => BinOp::Ge,
                _ => break,
            };
            self.advance();
            let right = self.parse_shift()?;
            left = Expr::binop(op, left, right);
        }
        Ok(left)
    }

    fn parse_shift(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_additive()?;
        loop {
            let op = match self.peek() {
                TokenKind::Shl => BinOp::Shl,
                TokenKind::Shr => BinOp::Shr,
                _ => break,
            };
            self.advance();
            let right = self.parse_additive()?;
            left = Expr::binop(op, left, right);
        }
        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                TokenKind::Plus => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative()?;
            left = Expr::binop(op, left, right);
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                TokenKind::Star => BinOp::Mul,
                TokenKind::Slash => BinOp::Div,
                TokenKind::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary()?;
            left = Expr::binop(op, left, right);
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        match self.peek() {
            TokenKind::Minus => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::unop(UnaryOp::Neg, e))
            }
            TokenKind::Bang => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::unop(UnaryOp::Not, e))
            }
            TokenKind::Tilde => {
                self.advance();
                let e = self.parse_unary()?;
                Ok(Expr::unop(UnaryOp::BitNot, e))
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        match self.peek().clone() {
            TokenKind::IntLit(v) => {
                self.advance();
                Ok(Expr::IntLiteral(v))
            }
            TokenKind::FloatLit(v) => {
                self.advance();
                Ok(Expr::FloatLiteral(v))
            }
            TokenKind::BoolLit(v) => {
                self.advance();
                Ok(Expr::BoolLiteral(v))
            }
            TokenKind::Pid => {
                self.advance();
                Ok(Expr::ProcessorId)
            }
            TokenKind::NumProcessors => {
                self.advance();
                Ok(Expr::NumProcessors)
            }
            TokenKind::SharedReadKw => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                let mem = self.parse_expr()?;
                self.expect(TokenKind::Comma)?;
                let idx = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                Ok(Expr::SharedRead(Box::new(mem), Box::new(idx)))
            }
            TokenKind::LParen => {
                self.advance();
                let e = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                Ok(e)
            }
            TokenKind::Ident(name) => {
                self.advance();
                // Function call?
                if self.check(&TokenKind::LParen) {
                    self.advance();
                    let mut args = Vec::new();
                    if !self.check(&TokenKind::RParen) {
                        loop {
                            args.push(self.parse_expr()?);
                            if self.check(&TokenKind::Comma) {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    // Map min/max to BinOp
                    if args.len() == 2 {
                        if name == "min" {
                            return Ok(Expr::binop(BinOp::Min, args.remove(0), args.remove(0)));
                        }
                        if name == "max" {
                            return Ok(Expr::binop(BinOp::Max, args.remove(0), args.remove(0)));
                        }
                    }
                    Ok(Expr::FunctionCall(name, args))
                } else {
                    Ok(Expr::Variable(name))
                }
            }
            ref tok => Err(self.err(format!("Expected expression, got {:?}", tok))),
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a PRAM IR program from its textual DSL representation.
pub fn parse_program(source: &str) -> Result<PramProgram, ParseError> {
    let tokens = Lexer::new(source).tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

/// Parse multiple PRAM programs from a single source string.
/// Programs are separated by being consecutive `algorithm ... { ... }` blocks.
pub fn parse_multiple(source: &str) -> Result<Vec<PramProgram>, ParseError> {
    let tokens = Lexer::new(source).tokenize()?;
    let mut parser = Parser::new(tokens);
    let mut programs = Vec::new();
    while !parser.check(&TokenKind::Eof) {
        programs.push(parser.parse_program()?);
    }
    if programs.is_empty() {
        return Err(ParseError::new("No programs found in input", 1, 1));
    }
    Ok(programs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_program() {
        let src = r#"
            algorithm noop() model EREW {
                processors = 1;
            }
        "#;
        let prog = parse_program(src).unwrap();
        assert_eq!(prog.name, "noop");
        assert_eq!(prog.memory_model, MemoryModel::EREW);
        assert!(prog.parameters.is_empty());
        assert!(prog.body.is_empty());
    }

    #[test]
    fn test_parse_with_params_and_shared() {
        let src = r#"
            algorithm init(n: i64) model CREW {
                shared A: i64[n];
                processors = n;
                parallel_for p in 0..n {
                    shared_write(A, pid, 0);
                }
            }
        "#;
        let prog = parse_program(src).unwrap();
        assert_eq!(prog.name, "init");
        assert_eq!(prog.memory_model, MemoryModel::CREW);
        assert_eq!(prog.parameters.len(), 1);
        assert_eq!(prog.shared_memory.len(), 1);
        assert_eq!(prog.shared_memory[0].name, "A");
        assert_eq!(prog.shared_memory[0].elem_type, PramType::Int64);
        assert_eq!(prog.body.len(), 1);
    }

    #[test]
    fn test_parse_expressions() {
        let src = r#"
            algorithm expr_test() model EREW {
                processors = 1;
                let x: i64 = 1 + 2 * 3;
                let y: i64 = (1 + 2) * 3;
                let z: bool = true && false || true;
            }
        "#;
        let prog = parse_program(src).unwrap();
        assert_eq!(prog.body.len(), 3);

        // First: x = 1 + (2*3) = 7
        if let Stmt::LocalDecl(_, _, Some(ref e)) = prog.body[0] {
            assert_eq!(e.eval_const_int(), Some(7));
        } else {
            panic!("expected LocalDecl");
        }

        // Second: y = (1+2)*3 = 9
        if let Stmt::LocalDecl(_, _, Some(ref e)) = prog.body[1] {
            assert_eq!(e.eval_const_int(), Some(9));
        } else {
            panic!("expected LocalDecl");
        }
    }

    #[test]
    fn test_parse_if_else() {
        let src = r#"
            algorithm iftest() model EREW {
                processors = 1;
                if true {
                    x = 1;
                } else {
                    x = 2;
                }
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::If { then_body, else_body, .. } => {
                assert_eq!(then_body.len(), 1);
                assert_eq!(else_body.len(), 1);
            }
            _ => panic!("expected If"),
        }
    }

    #[test]
    fn test_parse_seq_for() {
        let src = r#"
            algorithm loop_test() model EREW {
                processors = 1;
                for i in 0..10 {
                    x = i;
                }
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::SeqFor { var, .. } => assert_eq!(var, "i"),
            _ => panic!("expected SeqFor"),
        }
    }

    #[test]
    fn test_parse_while() {
        let src = r#"
            algorithm wt() model EREW {
                processors = 1;
                while x < 10 {
                    x = x + 1;
                }
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::While { .. } => {}
            _ => panic!("expected While"),
        }
    }

    #[test]
    fn test_parse_shared_read() {
        let src = r#"
            algorithm sr() model CREW {
                shared A: i64[8];
                processors = 8;
                parallel_for p in 0..8 {
                    let v: i64 = shared_read(A, pid);
                }
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::ParallelFor { body, .. } => {
                match &body[0] {
                    Stmt::LocalDecl(_, _, Some(Expr::SharedRead(_, _))) => {}
                    other => panic!("expected SharedRead init, got {:?}", other),
                }
            }
            _ => panic!("expected ParallelFor"),
        }
    }

    #[test]
    fn test_parse_barrier() {
        let src = r#"
            algorithm bar() model CREW {
                processors = 4;
                parallel_for p in 0..4 {
                    x = 1;
                    barrier;
                    y = 2;
                }
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::ParallelFor { body, .. } => {
                assert_eq!(body.len(), 3);
                assert_eq!(body[1], Stmt::Barrier);
            }
            _ => panic!("expected ParallelFor"),
        }
    }

    #[test]
    fn test_parse_return() {
        let src = r#"
            algorithm ret() model EREW {
                processors = 1;
                return 42;
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::Return(Some(Expr::IntLiteral(42))) => {}
            other => panic!("expected Return(42), got {:?}", other),
        }
    }

    #[test]
    fn test_parse_function_call() {
        let src = r#"
            algorithm fc() model EREW {
                processors = 1;
                let v: i64 = min(a, b);
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::LocalDecl(_, _, Some(Expr::BinOp(BinOp::Min, _, _))) => {}
            other => panic!("expected min BinOp, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_unary() {
        let src = r#"
            algorithm un() model EREW {
                processors = 1;
                let a: i64 = -5;
                let b: bool = !true;
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::LocalDecl(_, _, Some(Expr::UnaryOp(UnaryOp::Neg, _))) => {}
            other => panic!("expected Neg, got {:?}", other),
        }
        match &prog.body[1] {
            Stmt::LocalDecl(_, _, Some(Expr::UnaryOp(UnaryOp::Not, _))) => {}
            other => panic!("expected Not, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_crcw_model() {
        let src = r#"
            algorithm cw() model CRCW_Priority {
                processors = 1;
            }
        "#;
        let prog = parse_program(src).unwrap();
        assert_eq!(prog.memory_model, MemoryModel::CRCWPriority);
    }

    #[test]
    fn test_parse_comments_skipped() {
        let src = r#"
            // header comment
            algorithm cmt() model EREW {
                processors = 1;
                // body comment
                x = 1; // inline comment
            }
        "#;
        let prog = parse_program(src).unwrap();
        assert_eq!(prog.body.len(), 1);
    }

    #[test]
    fn test_parse_error_unexpected_char() {
        let src = "algorithm a() model EREW { @ }";
        assert!(parse_program(src).is_err());
    }

    #[test]
    fn test_parse_shared_type_with_wrapper() {
        let src = r#"
            algorithm sw() model EREW {
                shared A: shared<i64>[10];
                processors = 1;
            }
        "#;
        let prog = parse_program(src).unwrap();
        assert_eq!(prog.shared_memory[0].elem_type, PramType::Int64);
    }

    #[test]
    fn test_parse_else_if() {
        let src = r#"
            algorithm ei() model EREW {
                processors = 1;
                if x == 0 {
                    y = 1;
                } else if x == 1 {
                    y = 2;
                } else {
                    y = 3;
                }
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::If { else_body, .. } => {
                // else-if is desugared into an If inside the else body
                assert_eq!(else_body.len(), 1);
                match &else_body[0] {
                    Stmt::If { .. } => {}
                    _ => panic!("expected nested If"),
                }
            }
            _ => panic!("expected If"),
        }
    }

    #[test]
    fn test_parse_comparison_ops() {
        let src = r#"
            algorithm cmp() model EREW {
                processors = 1;
                let a: bool = 1 < 2;
                let b: bool = 3 >= 3;
                let c: bool = 4 != 5;
            }
        "#;
        let prog = parse_program(src).unwrap();
        assert_eq!(prog.body.len(), 3);
    }

    #[test]
    fn test_parse_block() {
        let src = r#"
            algorithm blk() model EREW {
                processors = 1;
                {
                    x = 1;
                    y = 2;
                }
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::Block(stmts) => assert_eq!(stmts.len(), 2),
            _ => panic!("expected Block"),
        }
    }

    #[test]
    fn test_parse_atomic_cas() {
        let src = r#"
            algorithm cas_test() model CRCW_Priority {
                shared A: i64[10];
                processors = 4;
                parallel_for p in 0..4 {
                    result = atomic_cas(A, pid, 0, 1);
                }
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::ParallelFor { body, .. } => {
                match &body[0] {
                    Stmt::AtomicCAS { result_var, .. } => {
                        assert_eq!(result_var, "result");
                    }
                    other => panic!("expected AtomicCAS, got {:?}", other),
                }
            }
            _ => panic!("expected ParallelFor"),
        }
    }

    #[test]
    fn test_parse_fetch_add() {
        let src = r#"
            algorithm fa_test() model CRCW_Priority {
                shared counter: i64[1];
                processors = 8;
                parallel_for p in 0..8 {
                    old = fetch_add(counter, 0, 1);
                }
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::ParallelFor { body, .. } => {
                match &body[0] {
                    Stmt::FetchAdd { result_var, .. } => {
                        assert_eq!(result_var, "old");
                    }
                    other => panic!("expected FetchAdd, got {:?}", other),
                }
            }
            _ => panic!("expected ParallelFor"),
        }
    }

    #[test]
    fn test_parse_prefix_sum() {
        let src = r#"
            algorithm ps_test() model CREW {
                shared A: i64[8];
                shared B: i64[8];
                processors = 8;
                prefix_sum(A, B, 8, +);
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::PrefixSum { input, output, op, .. } => {
                assert_eq!(input, "A");
                assert_eq!(output, "B");
                assert_eq!(*op, BinOp::Add);
            }
            other => panic!("expected PrefixSum, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_assert() {
        let src = r#"
            algorithm assert_test() model EREW {
                processors = 1;
                assert(true, "should hold");
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::Assert(_, msg) => assert_eq!(msg, "should hold"),
            other => panic!("expected Assert, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_nop() {
        let src = r#"
            algorithm nop_test() model EREW {
                processors = 1;
                nop;
            }
        "#;
        let prog = parse_program(src).unwrap();
        assert_eq!(prog.body[0], Stmt::Nop);
    }

    #[test]
    fn test_parse_multiple_programs() {
        let src = r#"
            algorithm first() model EREW {
                processors = 1;
            }
            algorithm second() model CREW {
                processors = 2;
            }
        "#;
        let progs = parse_multiple(src).unwrap();
        assert_eq!(progs.len(), 2);
        assert_eq!(progs[0].name, "first");
        assert_eq!(progs[1].name, "second");
    }

    #[test]
    fn test_parse_error_with_context() {
        let err = ParseError::new("bad token", 1, 5).with_context("expression parsing");
        assert!(err.message.contains("expression parsing"));
        assert!(err.message.contains("bad token"));
    }

    #[test]
    fn test_parse_atomic_cas_standalone() {
        let src = r#"
            algorithm cas2() model CRCW_Priority {
                shared A: i64[10];
                processors = 4;
                atomic_cas(A, 0, 0, 1);
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::AtomicCAS { result_var, .. } => assert_eq!(result_var, "_"),
            other => panic!("expected AtomicCAS, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_fetch_add_standalone() {
        let src = r#"
            algorithm fa2() model CRCW_Priority {
                shared A: i64[1];
                processors = 4;
                fetch_add(A, 0, 1);
            }
        "#;
        let prog = parse_program(src).unwrap();
        match &prog.body[0] {
            Stmt::FetchAdd { result_var, .. } => assert_eq!(result_var, "_"),
            other => panic!("expected FetchAdd, got {:?}", other),
        }
    }
}
