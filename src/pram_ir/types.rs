use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents the primitive data types in our PRAM IR.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PramType {
    /// 64-bit signed integer
    Int64,
    /// 32-bit signed integer
    Int32,
    /// 64-bit floating point
    Float64,
    /// 32-bit floating point
    Float32,
    /// Boolean
    Bool,
    /// Fixed-size array of a given type
    Array(Box<PramType>, usize),
    /// Dynamically-sized shared memory region
    SharedMemory(Box<PramType>),
    /// Processor ID type (bounded by number of processors)
    ProcessorId,
    /// Pointer/reference to a type in shared memory
    SharedRef(Box<PramType>),
    /// Tuple of types
    Tuple(Vec<PramType>),
    /// Void / unit type
    Unit,
    /// A struct type with named fields
    Struct(String, Vec<(String, PramType)>),
}

impl PramType {
    /// Returns the size in bytes of this type (for memory layout).
    pub fn size_bytes(&self) -> usize {
        match self {
            PramType::Int64 => 8,
            PramType::Int32 => 4,
            PramType::Float64 => 8,
            PramType::Float32 => 4,
            PramType::Bool => 1,
            PramType::Array(inner, count) => inner.size_bytes() * count,
            PramType::SharedMemory(inner) => inner.size_bytes(),
            PramType::ProcessorId => 8,
            PramType::SharedRef(_) => 8,
            PramType::Tuple(types) => types.iter().map(|t| t.size_bytes()).sum(),
            PramType::Unit => 0,
            PramType::Struct(_, fields) => fields.iter().map(|(_, t)| t.size_bytes()).sum(),
        }
    }

    /// Whether this type can be stored in shared memory.
    pub fn is_shareable(&self) -> bool {
        match self {
            PramType::Int64 | PramType::Int32 | PramType::Float64 | PramType::Float32 => true,
            PramType::Bool => true,
            PramType::Array(inner, _) => inner.is_shareable(),
            PramType::SharedMemory(_) => true,
            PramType::ProcessorId => true,
            PramType::SharedRef(_) => true,
            PramType::Tuple(types) => types.iter().all(|t| t.is_shareable()),
            PramType::Unit => false,
            PramType::Struct(_, fields) => fields.iter().all(|(_, t)| t.is_shareable()),
        }
    }

    /// Whether this type is numeric.
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            PramType::Int64 | PramType::Int32 | PramType::Float64 | PramType::Float32
        )
    }

    /// Whether this type is an integer type.
    pub fn is_integer(&self) -> bool {
        matches!(self, PramType::Int64 | PramType::Int32)
    }

    /// Returns the element type if this is an array or shared memory.
    pub fn element_type(&self) -> Option<&PramType> {
        match self {
            PramType::Array(inner, _) => Some(inner),
            PramType::SharedMemory(inner) => Some(inner),
            PramType::SharedRef(inner) => Some(inner),
            _ => None,
        }
    }

    /// Alignment requirement in bytes.
    pub fn alignment(&self) -> usize {
        match self {
            PramType::Int64 | PramType::Float64 | PramType::ProcessorId | PramType::SharedRef(_) => 8,
            PramType::Int32 | PramType::Float32 => 4,
            PramType::Bool => 1,
            PramType::Array(inner, _) => inner.alignment(),
            PramType::SharedMemory(inner) => inner.alignment(),
            PramType::Tuple(types) => types.iter().map(|t| t.alignment()).max().unwrap_or(1),
            PramType::Unit => 1,
            PramType::Struct(_, fields) => {
                fields.iter().map(|(_, t)| t.alignment()).max().unwrap_or(1)
            }
        }
    }

    /// Maximum representable value for integer types, or `None` for non-integer types.
    pub fn max_value(&self) -> Option<i64> {
        match self {
            PramType::Int64 => Some(i64::MAX),
            PramType::Int32 => Some(i32::MAX as i64),
            PramType::Bool => Some(1),
            _ => None,
        }
    }

    /// Minimum representable value for integer types, or `None` for non-integer types.
    pub fn min_value(&self) -> Option<i64> {
        match self {
            PramType::Int64 => Some(i64::MIN),
            PramType::Int32 => Some(i32::MIN as i64),
            PramType::Bool => Some(0),
            _ => None,
        }
    }

    /// Check if two types are compatible for assignment.
    pub fn is_assignable_from(&self, other: &PramType) -> bool {
        if self == other {
            return true;
        }
        match (self, other) {
            (PramType::Int64, PramType::Int32) => true,
            (PramType::Float64, PramType::Float32) => true,
            (PramType::Float64, PramType::Int64) => true,
            (PramType::Float64, PramType::Int32) => true,
            (PramType::Float32, PramType::Int32) => true,
            _ => false,
        }
    }
}

impl fmt::Display for PramType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PramType::Int64 => write!(f, "i64"),
            PramType::Int32 => write!(f, "i32"),
            PramType::Float64 => write!(f, "f64"),
            PramType::Float32 => write!(f, "f32"),
            PramType::Bool => write!(f, "bool"),
            PramType::Array(inner, size) => write!(f, "[{}; {}]", inner, size),
            PramType::SharedMemory(inner) => write!(f, "shared<{}>", inner),
            PramType::ProcessorId => write!(f, "pid"),
            PramType::SharedRef(inner) => write!(f, "&shared {}", inner),
            PramType::Tuple(types) => {
                write!(f, "(")?;
                for (i, t) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            }
            PramType::Unit => write!(f, "()"),
            PramType::Struct(name, _) => write!(f, "struct {}", name),
        }
    }
}

/// Type environment for checking PRAM programs.
#[derive(Debug, Clone, Default)]
pub struct TypeEnv {
    pub bindings: Vec<(String, PramType)>,
    pub shared_regions: Vec<(String, PramType, usize)>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
            shared_regions: Vec::new(),
        }
    }

    pub fn bind(&mut self, name: String, ty: PramType) {
        self.bindings.push((name, ty));
    }

    pub fn lookup(&self, name: &str) -> Option<&PramType> {
        self.bindings
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t)
    }

    pub fn declare_shared(&mut self, name: String, elem_type: PramType, size: usize) {
        self.shared_regions.push((name.clone(), elem_type.clone(), size));
        self.bind(
            name,
            PramType::SharedMemory(Box::new(elem_type)),
        );
    }

    pub fn get_shared_region(&self, name: &str) -> Option<(&PramType, usize)> {
        self.shared_regions
            .iter()
            .rev()
            .find(|(n, _, _)| n == name)
            .map(|(_, t, s)| (t, *s))
    }

    /// Push a scope (for lexical scoping in parallel_for bodies, etc.)
    pub fn push_scope(&self) -> TypeEnv {
        self.clone()
    }

    /// Return the current scope depth (number of bindings, used as a proxy).
    pub fn scope_depth(&self) -> usize {
        self.bindings.len()
    }

    /// Return all current bindings as a slice.
    pub fn all_bindings(&self) -> &[(String, PramType)] {
        &self.bindings
    }

    /// Return all shared region declarations as a slice.
    pub fn all_shared_regions(&self) -> &[(String, PramType, usize)] {
        &self.shared_regions
    }
}

/// Describes the compatibility between two types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeCompatibility {
    /// Types are exactly the same.
    Identical,
    /// Source can be widened into target without loss (e.g. i32 -> i64).
    Widening,
    /// Source can be narrowed into target with potential loss (e.g. i64 -> i32).
    Narrowing,
    /// Types are incompatible.
    Incompatible,
}

/// Check compatibility when assigning a value of type `source` to a location of type `target`.
pub fn check_compatibility(target: &PramType, source: &PramType) -> TypeCompatibility {
    if target == source {
        return TypeCompatibility::Identical;
    }
    match (target, source) {
        // Widening integer
        (PramType::Int64, PramType::Int32) => TypeCompatibility::Widening,
        // Widening float
        (PramType::Float64, PramType::Float32) => TypeCompatibility::Widening,
        // Int to float widening
        (PramType::Float64, PramType::Int64) => TypeCompatibility::Widening,
        (PramType::Float64, PramType::Int32) => TypeCompatibility::Widening,
        (PramType::Float32, PramType::Int32) => TypeCompatibility::Widening,
        // Narrowing
        (PramType::Int32, PramType::Int64) => TypeCompatibility::Narrowing,
        (PramType::Float32, PramType::Float64) => TypeCompatibility::Narrowing,
        (PramType::Int64, PramType::Float64) => TypeCompatibility::Narrowing,
        (PramType::Int32, PramType::Float64) => TypeCompatibility::Narrowing,
        (PramType::Int32, PramType::Float32) => TypeCompatibility::Narrowing,
        (PramType::Int64, PramType::Float32) => TypeCompatibility::Narrowing,
        (PramType::Float32, PramType::Int64) => TypeCompatibility::Widening,
        // Array element compatibility
        (PramType::Array(t1, s1), PramType::Array(t2, s2)) => {
            if s1 != s2 {
                return TypeCompatibility::Incompatible;
            }
            check_compatibility(t1, t2)
        }
        // SharedMemory element compatibility
        (PramType::SharedMemory(t1), PramType::SharedMemory(t2)) => {
            check_compatibility(t1, t2)
        }
        _ => TypeCompatibility::Incompatible,
    }
}

/// Represents a type-checking error.
#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub location: Option<SourceLocation>,
}

impl TypeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            location: None,
        }
    }

    pub fn with_location(mut self, loc: SourceLocation) -> Self {
        self.location = Some(loc);
        self
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref loc) = self.location {
            write!(f, "Type error at {}: {}", loc, self.message)
        } else {
            write!(f, "Type error: {}", self.message)
        }
    }
}

/// Source location for error reporting.
#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub line: usize,
    pub column: usize,
    pub file: Option<String>,
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref file) = self.file {
            write!(f, "{}:{}:{}", file, self.line, self.column)
        } else {
            write!(f, "{}:{}", self.line, self.column)
        }
    }
}

/// Type-check an expression, returning its type or a type error.
pub fn typecheck_expr(
    env: &TypeEnv,
    expr: &crate::pram_ir::ast::Expr,
) -> Result<PramType, TypeError> {
    use crate::pram_ir::ast::Expr;
    match expr {
        Expr::IntLiteral(_) => Ok(PramType::Int64),
        Expr::FloatLiteral(_) => Ok(PramType::Float64),
        Expr::BoolLiteral(_) => Ok(PramType::Bool),
        Expr::Variable(name) => env
            .lookup(name)
            .cloned()
            .ok_or_else(|| TypeError::new(format!("Undefined variable: {}", name))),
        Expr::ProcessorId => Ok(PramType::ProcessorId),
        Expr::NumProcessors => Ok(PramType::Int64),
        Expr::BinOp(op, left, right) => {
            let lt = typecheck_expr(env, left)?;
            let rt = typecheck_expr(env, right)?;
            use crate::pram_ir::ast::BinOp;
            match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                    if lt.is_numeric() && rt.is_numeric() {
                        if lt == PramType::Float64 || rt == PramType::Float64 {
                            Ok(PramType::Float64)
                        } else if lt == PramType::Int64 || rt == PramType::Int64 {
                            Ok(PramType::Int64)
                        } else {
                            Ok(PramType::Int32)
                        }
                    } else if (lt == PramType::ProcessorId || lt.is_integer())
                        && (rt == PramType::ProcessorId || rt.is_integer())
                    {
                        Ok(PramType::Int64)
                    } else {
                        Err(TypeError::new(format!(
                            "Cannot apply {:?} to {} and {}",
                            op, lt, rt
                        )))
                    }
                }
                BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne => {
                    Ok(PramType::Bool)
                }
                BinOp::And | BinOp::Or => {
                    if lt == PramType::Bool && rt == PramType::Bool {
                        Ok(PramType::Bool)
                    } else {
                        Err(TypeError::new(format!(
                            "Boolean operators require bool, got {} and {}",
                            lt, rt
                        )))
                    }
                }
                BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                    if lt.is_integer() && rt.is_integer() {
                        Ok(PramType::Int64)
                    } else {
                        Err(TypeError::new(format!(
                            "Bitwise operators require integers, got {} and {}",
                            lt, rt
                        )))
                    }
                }
                BinOp::Min | BinOp::Max => {
                    if lt.is_numeric() && rt.is_numeric() {
                        if lt == PramType::Float64 || rt == PramType::Float64 {
                            Ok(PramType::Float64)
                        } else {
                            Ok(PramType::Int64)
                        }
                    } else {
                        Err(TypeError::new(format!(
                            "Min/Max require numeric types, got {} and {}",
                            lt, rt
                        )))
                    }
                }
            }
        }
        Expr::UnaryOp(op, operand) => {
            let t = typecheck_expr(env, operand)?;
            use crate::pram_ir::ast::UnaryOp;
            match op {
                UnaryOp::Neg => {
                    if t.is_numeric() || t == PramType::ProcessorId {
                        Ok(t)
                    } else {
                        Err(TypeError::new(format!("Cannot negate {}", t)))
                    }
                }
                UnaryOp::Not => {
                    if t == PramType::Bool {
                        Ok(PramType::Bool)
                    } else {
                        Err(TypeError::new(format!("Cannot negate non-bool {}", t)))
                    }
                }
                UnaryOp::BitNot => {
                    if t.is_integer() {
                        Ok(PramType::Int64)
                    } else {
                        Err(TypeError::new(format!("Bitwise not requires integer, got {}", t)))
                    }
                }
            }
        }
        Expr::SharedRead(mem, index) => {
            let mt = typecheck_expr(env, mem)?;
            let _it = typecheck_expr(env, index)?;
            match mt {
                PramType::SharedMemory(inner) => Ok(*inner),
                PramType::Array(inner, _) => Ok(*inner),
                _ => Err(TypeError::new(format!(
                    "Cannot read from non-shared type: {}",
                    mt
                ))),
            }
        }
        Expr::ArrayIndex(arr, index) => {
            let at = typecheck_expr(env, arr)?;
            let _it = typecheck_expr(env, index)?;
            match at {
                PramType::Array(inner, _) => Ok(*inner),
                PramType::SharedMemory(inner) => Ok(*inner),
                _ => Err(TypeError::new(format!(
                    "Cannot index non-array type: {}",
                    at
                ))),
            }
        }
        Expr::FunctionCall(name, args) => {
            for arg in args {
                let _t = typecheck_expr(env, arg)?;
            }
            match name.as_str() {
                "log2" | "sqrt" | "abs" => Ok(PramType::Float64),
                "min" | "max" => Ok(PramType::Int64),
                "len" => Ok(PramType::Int64),
                _ => Ok(PramType::Int64),
            }
        }
        Expr::Cast(inner, target_type) => {
            let _src = typecheck_expr(env, inner)?;
            Ok(target_type.clone())
        }
        Expr::Conditional(cond, then_expr, else_expr) => {
            let ct = typecheck_expr(env, cond)?;
            if ct != PramType::Bool {
                return Err(TypeError::new(format!(
                    "Condition must be bool, got {}",
                    ct
                )));
            }
            let tt = typecheck_expr(env, then_expr)?;
            let _et = typecheck_expr(env, else_expr)?;
            Ok(tt)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_size() {
        assert_eq!(PramType::Int64.size_bytes(), 8);
        assert_eq!(PramType::Int32.size_bytes(), 4);
        assert_eq!(PramType::Float64.size_bytes(), 8);
        assert_eq!(PramType::Bool.size_bytes(), 1);
        assert_eq!(
            PramType::Array(Box::new(PramType::Int64), 10).size_bytes(),
            80
        );
    }

    #[test]
    fn test_type_shareable() {
        assert!(PramType::Int64.is_shareable());
        assert!(PramType::Float64.is_shareable());
        assert!(!PramType::Unit.is_shareable());
        assert!(PramType::Array(Box::new(PramType::Int32), 5).is_shareable());
    }

    #[test]
    fn test_type_env_binding() {
        let mut env = TypeEnv::new();
        env.bind("x".to_string(), PramType::Int64);
        assert_eq!(env.lookup("x"), Some(&PramType::Int64));
        assert_eq!(env.lookup("y"), None);
    }

    #[test]
    fn test_type_env_shared() {
        let mut env = TypeEnv::new();
        env.declare_shared("A".to_string(), PramType::Int64, 1000);
        let (ty, size) = env.get_shared_region("A").unwrap();
        assert_eq!(*ty, PramType::Int64);
        assert_eq!(size, 1000);
    }

    #[test]
    fn test_type_display() {
        assert_eq!(format!("{}", PramType::Int64), "i64");
        assert_eq!(
            format!("{}", PramType::SharedMemory(Box::new(PramType::Int32))),
            "shared<i32>"
        );
        assert_eq!(
            format!("{}", PramType::Array(Box::new(PramType::Float64), 3)),
            "[f64; 3]"
        );
    }

    #[test]
    fn test_assignable() {
        assert!(PramType::Int64.is_assignable_from(&PramType::Int32));
        assert!(PramType::Float64.is_assignable_from(&PramType::Int64));
        assert!(!PramType::Int32.is_assignable_from(&PramType::Int64));
        assert!(PramType::Int64.is_assignable_from(&PramType::Int64));
    }

    #[test]
    fn test_alignment() {
        assert_eq!(PramType::Int64.alignment(), 8);
        assert_eq!(PramType::Int32.alignment(), 4);
        assert_eq!(PramType::Bool.alignment(), 1);
    }

    #[test]
    fn test_element_type() {
        let arr = PramType::Array(Box::new(PramType::Int64), 10);
        assert_eq!(arr.element_type(), Some(&PramType::Int64));
        assert_eq!(PramType::Bool.element_type(), None);
    }

    #[test]
    fn test_max_min_value() {
        assert_eq!(PramType::Int64.max_value(), Some(i64::MAX));
        assert_eq!(PramType::Int64.min_value(), Some(i64::MIN));
        assert_eq!(PramType::Int32.max_value(), Some(i32::MAX as i64));
        assert_eq!(PramType::Int32.min_value(), Some(i32::MIN as i64));
        assert_eq!(PramType::Bool.max_value(), Some(1));
        assert_eq!(PramType::Bool.min_value(), Some(0));
        assert_eq!(PramType::Float64.max_value(), None);
    }

    #[test]
    fn test_type_compatibility_identical() {
        assert_eq!(
            check_compatibility(&PramType::Int64, &PramType::Int64),
            TypeCompatibility::Identical
        );
    }

    #[test]
    fn test_type_compatibility_widening() {
        assert_eq!(
            check_compatibility(&PramType::Int64, &PramType::Int32),
            TypeCompatibility::Widening
        );
        assert_eq!(
            check_compatibility(&PramType::Float64, &PramType::Float32),
            TypeCompatibility::Widening
        );
        assert_eq!(
            check_compatibility(&PramType::Float64, &PramType::Int64),
            TypeCompatibility::Widening
        );
    }

    #[test]
    fn test_type_compatibility_narrowing() {
        assert_eq!(
            check_compatibility(&PramType::Int32, &PramType::Int64),
            TypeCompatibility::Narrowing
        );
        assert_eq!(
            check_compatibility(&PramType::Float32, &PramType::Float64),
            TypeCompatibility::Narrowing
        );
    }

    #[test]
    fn test_type_compatibility_incompatible() {
        assert_eq!(
            check_compatibility(&PramType::Bool, &PramType::Int64),
            TypeCompatibility::Incompatible
        );
        assert_eq!(
            check_compatibility(&PramType::Unit, &PramType::Int64),
            TypeCompatibility::Incompatible
        );
    }

    #[test]
    fn test_scope_depth() {
        let mut env = TypeEnv::new();
        assert_eq!(env.scope_depth(), 0);
        env.bind("x".to_string(), PramType::Int64);
        assert_eq!(env.scope_depth(), 1);
        env.bind("y".to_string(), PramType::Bool);
        assert_eq!(env.scope_depth(), 2);
    }

    #[test]
    fn test_all_bindings() {
        let mut env = TypeEnv::new();
        env.bind("x".to_string(), PramType::Int64);
        env.bind("y".to_string(), PramType::Bool);
        let bindings = env.all_bindings();
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0].0, "x");
        assert_eq!(bindings[1].0, "y");
    }

    #[test]
    fn test_all_shared_regions() {
        let mut env = TypeEnv::new();
        env.declare_shared("A".to_string(), PramType::Int64, 100);
        let regions = env.all_shared_regions();
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].0, "A");
        assert_eq!(regions[0].2, 100);
    }

    #[test]
    fn test_type_compatibility_arrays() {
        let a1 = PramType::Array(Box::new(PramType::Int32), 10);
        let a2 = PramType::Array(Box::new(PramType::Int32), 10);
        assert_eq!(check_compatibility(&a1, &a2), TypeCompatibility::Identical);

        let a3 = PramType::Array(Box::new(PramType::Int64), 10);
        assert_eq!(check_compatibility(&a3, &a1), TypeCompatibility::Widening);

        let a4 = PramType::Array(Box::new(PramType::Int32), 5);
        assert_eq!(check_compatibility(&a1, &a4), TypeCompatibility::Incompatible);
    }
}
