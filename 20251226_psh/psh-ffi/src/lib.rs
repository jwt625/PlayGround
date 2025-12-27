//! FFI layer for psh-core
//!
//! This module provides C-compatible bindings for Swift integration.
//! All functions are designed to be called from Swift via a C bridge.

#![allow(missing_docs)]

use psh_core::{ExpandResult, Expander, SnippetCollection, Warning};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::Mutex;

/// Global expander instance
static EXPANDER: Mutex<Option<Expander>> = Mutex::new(None);

/// C-compatible string (null-terminated)
pub type PshCString = *mut c_char;

/// C-compatible warning type
#[repr(C)]
pub struct PshWarning {
    /// Warning type: 0 = `UnknownNamespace`, 1 = `UnknownOperation`, 2 = `UnknownKey`
    pub warning_type: u32,
    /// The namespace (or empty string if not applicable)
    pub namespace: PshCString,
    /// The operation or key name (or empty string if not applicable)
    pub name: PshCString,
}

/// C-compatible expansion result
#[repr(C)]
pub struct PshExpandResult {
    /// The expanded text (null if error)
    pub text: PshCString,
    /// Array of warnings
    pub warnings: *mut PshWarning,
    /// Number of warnings
    pub warning_count: usize,
    /// Error message (null if success)
    pub error: PshCString,
}

/// Initialize the psh engine with snippets from a file
///
/// # Safety
///
/// `snippets_path` must be a valid null-terminated C string
///
/// # Panics
///
/// Panics if the global mutex is poisoned
#[no_mangle]
pub unsafe extern "C" fn psh_init(snippets_path: *const c_char) -> bool {
    if snippets_path.is_null() {
        return false;
    }

    let Ok(path_cstr) = CStr::from_ptr(snippets_path).to_str() else {
        return false;
    };

    let mut collection = SnippetCollection::new();
    if collection
        .load_from_file(std::path::Path::new(path_cstr))
        .is_err()
    {
        return false;
    }

    let expander = Expander::new(collection);
    *EXPANDER.lock().unwrap() = Some(expander);
    true
}

/// Expand text containing psh directives
///
/// # Safety
///
/// `text` must be a valid null-terminated C string.
/// Caller must free the returned `PshExpandResult` using `psh_free_result`.
///
/// # Panics
///
/// Panics if the global mutex is poisoned
#[no_mangle]
pub unsafe extern "C" fn psh_expand(text: *const c_char) -> *mut PshExpandResult {
    if text.is_null() {
        return ptr::null_mut();
    }

    let Ok(text_str) = CStr::from_ptr(text).to_str() else {
        return create_error_result("Invalid UTF-8 in input text");
    };

    let Some(ref expander) = *EXPANDER.lock().unwrap() else {
        return create_error_result("Expander not initialized. Call psh_init first.");
    };

    match expander.expand(text_str) {
        Ok(result) => create_success_result(result),
        Err(e) => create_error_result(&format!("Expansion error: {e}")),
    }
}

/// Free a `PshExpandResult` returned by `psh_expand`
///
/// # Safety
///
/// `result` must be a valid pointer returned by `psh_expand` and not already freed
#[no_mangle]
pub unsafe extern "C" fn psh_free_result(result: *mut PshExpandResult) {
    if result.is_null() {
        return;
    }

    let result_box = Box::from_raw(result);

    // Free text string
    if !result_box.text.is_null() {
        drop(CString::from_raw(result_box.text));
    }

    // Free error string
    if !result_box.error.is_null() {
        drop(CString::from_raw(result_box.error));
    }

    // Free warnings
    if !result_box.warnings.is_null() {
        let warnings_slice =
            std::slice::from_raw_parts_mut(result_box.warnings, result_box.warning_count);
        for warning in warnings_slice {
            if !warning.namespace.is_null() {
                drop(CString::from_raw(warning.namespace));
            }
            if !warning.name.is_null() {
                drop(CString::from_raw(warning.name));
            }
        }
        drop(Vec::from_raw_parts(
            result_box.warnings,
            result_box.warning_count,
            result_box.warning_count,
        ));
    }
}

/// Reload snippets from the same file path
///
/// # Safety
///
/// `snippets_path` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn psh_reload_snippets(snippets_path: *const c_char) -> bool {
    psh_init(snippets_path)
}

/// Shutdown the psh engine and free resources
///
/// # Panics
///
/// Panics if the global mutex is poisoned
#[no_mangle]
pub extern "C" fn psh_shutdown() {
    *EXPANDER.lock().unwrap() = None;
}

// Helper functions

/// Create a success result from an `ExpandResult`
fn create_success_result(result: ExpandResult) -> *mut PshExpandResult {
    let text = CString::new(result.text).map_or(ptr::null_mut(), CString::into_raw);

    let warnings = convert_warnings(&result.warnings);
    let warning_count = result.warnings.len();

    Box::into_raw(Box::new(PshExpandResult {
        text,
        warnings,
        warning_count,
        error: ptr::null_mut(),
    }))
}

/// Create an error result with an error message
fn create_error_result(error_msg: &str) -> *mut PshExpandResult {
    let error = CString::new(error_msg).map_or(ptr::null_mut(), CString::into_raw);

    Box::into_raw(Box::new(PshExpandResult {
        text: ptr::null_mut(),
        warnings: ptr::null_mut(),
        warning_count: 0,
        error,
    }))
}

/// Convert Rust warnings to C-compatible warnings
fn convert_warnings(warnings: &[Warning]) -> *mut PshWarning {
    if warnings.is_empty() {
        return ptr::null_mut();
    }

    let mut c_warnings = Vec::with_capacity(warnings.len());

    for warning in warnings {
        let (warning_type, namespace, name) = match warning {
            Warning::UnknownNamespace(ns) => {
                let ns_cstr = CString::new(ns.as_str()).unwrap_or_default();
                (0, ns_cstr.into_raw(), ptr::null_mut())
            }
            Warning::UnknownOperation { namespace: ns, op } => {
                let ns_cstr = CString::new(ns.as_str()).unwrap_or_default();
                let op_cstr = CString::new(op.as_str()).unwrap_or_default();
                (1, ns_cstr.into_raw(), op_cstr.into_raw())
            }
            Warning::UnknownKey { namespace: ns, key } => {
                let ns_cstr = CString::new(ns.as_str()).unwrap_or_default();
                let key_cstr = CString::new(key.as_str()).unwrap_or_default();
                (2, ns_cstr.into_raw(), key_cstr.into_raw())
            }
        };

        c_warnings.push(PshWarning {
            warning_type,
            namespace,
            name,
        });
    }

    let ptr = c_warnings.as_mut_ptr();
    std::mem::forget(c_warnings);
    ptr
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::sync::Mutex;

    // Test mutex to ensure tests run serially (they share global EXPANDER state)
    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_init_and_expand() {
        let _lock = TEST_MUTEX.lock().unwrap();

        // Create a test snippets file
        let test_toml = r#"
[[snippet]]
id = "test"
namespace = "t"
template = "Test {{ value }}"

[snippet.ops.base]
value = "default"
"#;

        let temp_file = std::env::temp_dir().join("test_snippets.toml");
        std::fs::write(&temp_file, test_toml).unwrap();

        unsafe {
            let path = CString::new(temp_file.to_str().unwrap()).unwrap();
            assert!(psh_init(path.as_ptr()));

            let text = CString::new("Hello ;;t world").unwrap();
            let result = psh_expand(text.as_ptr());
            assert!(!result.is_null());

            let result_ref = &*result;
            assert!(!result_ref.text.is_null());
            assert!(result_ref.error.is_null());

            let expanded = CStr::from_ptr(result_ref.text).to_str().unwrap();
            assert!(expanded.contains("Test default"));

            psh_free_result(result);
            psh_shutdown();
        }

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_expand_with_warnings() {
        let _lock = TEST_MUTEX.lock().unwrap();

        let test_toml = r#"
[[snippet]]
id = "test"
namespace = "t"
template = "Test"
"#;

        let temp_file = std::env::temp_dir().join("test_warnings.toml");
        std::fs::write(&temp_file, test_toml).unwrap();

        unsafe {
            let path = CString::new(temp_file.to_str().unwrap()).unwrap();
            assert!(psh_init(path.as_ptr()));

            let text = CString::new(";;unknown").unwrap();
            let result = psh_expand(text.as_ptr());
            assert!(!result.is_null());

            let result_ref = &*result;
            assert_eq!(result_ref.warning_count, 1);
            assert!(!result_ref.warnings.is_null());

            let warning = &*result_ref.warnings;
            assert_eq!(warning.warning_type, 0); // UnknownNamespace

            psh_free_result(result);
            psh_shutdown();
        }

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_null_pointer_handling() {
        let _lock = TEST_MUTEX.lock().unwrap();

        unsafe {
            // Test psh_init with null pointer
            assert!(!psh_init(ptr::null()));

            // Test psh_expand with null pointer (should return null)
            let result = psh_expand(ptr::null());
            assert!(result.is_null());

            // Test psh_free_result with null pointer (should not crash)
            psh_free_result(ptr::null_mut());
        }
    }

    #[test]
    fn test_expand_before_init() {
        let _lock = TEST_MUTEX.lock().unwrap();

        unsafe {
            psh_shutdown(); // Ensure clean state

            let text = CString::new(";;test").unwrap();
            let result = psh_expand(text.as_ptr());
            assert!(!result.is_null());

            let result_ref = &*result;
            assert!(result_ref.text.is_null());
            assert!(!result_ref.error.is_null());

            let error = CStr::from_ptr(result_ref.error).to_str().unwrap();
            assert!(error.contains("not initialized"));

            psh_free_result(result);
        }
    }

    #[test]
    fn test_reload_snippets() {
        let _lock = TEST_MUTEX.lock().unwrap();

        let test_toml_v1 = r#"
[[snippet]]
id = "test"
namespace = "t"
template = "Version 1"
"#;

        let test_toml_v2 = r#"
[[snippet]]
id = "test"
namespace = "t"
template = "Version 2"
"#;

        let temp_file = std::env::temp_dir().join("test_reload.toml");
        std::fs::write(&temp_file, test_toml_v1).unwrap();

        unsafe {
            let path = CString::new(temp_file.to_str().unwrap()).unwrap();
            assert!(psh_init(path.as_ptr()));

            let text = CString::new(";;t").unwrap();
            let result = psh_expand(text.as_ptr());
            let expanded = CStr::from_ptr((*result).text).to_str().unwrap();
            assert!(expanded.contains("Version 1"));
            psh_free_result(result);

            // Reload with new content
            std::fs::write(&temp_file, test_toml_v2).unwrap();
            assert!(psh_reload_snippets(path.as_ptr()));

            let result = psh_expand(text.as_ptr());
            let expanded = CStr::from_ptr((*result).text).to_str().unwrap();
            assert!(expanded.contains("Version 2"));
            psh_free_result(result);

            psh_shutdown();
        }

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_multiple_warning_types() {
        let _lock = TEST_MUTEX.lock().unwrap();

        let test_toml = r#"
[[snippet]]
id = "test"
namespace = "t"
template = "Test {{ value }}"

[snippet.ops.base]
value = "default"
"#;

        let temp_file = std::env::temp_dir().join("test_multi_warnings.toml");
        std::fs::write(&temp_file, test_toml).unwrap();

        unsafe {
            let path = CString::new(temp_file.to_str().unwrap()).unwrap();
            assert!(psh_init(path.as_ptr()));

            // Test with unknown namespace, unknown op, and unknown key
            let text = CString::new(";;unknown,badop,badkey=value").unwrap();
            let result = psh_expand(text.as_ptr());
            assert!(!result.is_null());

            let result_ref = &*result;
            assert_eq!(result_ref.warning_count, 1); // Unknown namespace

            let warning = &*result_ref.warnings;
            assert_eq!(warning.warning_type, 0); // UnknownNamespace

            psh_free_result(result);

            // Test with valid namespace but unknown op
            let text = CString::new(";;t,unknown_op").unwrap();
            let result = psh_expand(text.as_ptr());
            let result_ref = &*result;
            assert_eq!(result_ref.warning_count, 1);

            let warning = &*result_ref.warnings;
            assert_eq!(warning.warning_type, 1); // UnknownOperation

            psh_free_result(result);

            // Test with valid namespace but unknown key
            let text = CString::new(";;t,base,unknown_key=value").unwrap();
            let result = psh_expand(text.as_ptr());
            let result_ref = &*result;
            assert_eq!(result_ref.warning_count, 1);

            let warning = &*result_ref.warnings;
            assert_eq!(warning.warning_type, 2); // UnknownKey

            psh_free_result(result);
            psh_shutdown();
        }

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_invalid_file_path() {
        let _lock = TEST_MUTEX.lock().unwrap();

        unsafe {
            let path = CString::new("/nonexistent/path/to/snippets.toml").unwrap();
            assert!(!psh_init(path.as_ptr()));
        }
    }

    #[test]
    fn test_multiple_directives() {
        let _lock = TEST_MUTEX.lock().unwrap();

        let test_toml = r#"
[[snippet]]
id = "test1"
namespace = "t1"
template = "First"

[[snippet]]
id = "test2"
namespace = "t2"
template = "Second"
"#;

        let temp_file = std::env::temp_dir().join("test_multiple.toml");
        std::fs::write(&temp_file, test_toml).unwrap();

        unsafe {
            let path = CString::new(temp_file.to_str().unwrap()).unwrap();
            assert!(psh_init(path.as_ptr()));

            let text = CString::new("Start ;;t1 middle ;;t2 end").unwrap();
            let result = psh_expand(text.as_ptr());
            assert!(!result.is_null());

            let result_ref = &*result;
            assert!(!result_ref.text.is_null());
            assert!(result_ref.error.is_null());

            let expanded = CStr::from_ptr(result_ref.text).to_str().unwrap();
            assert!(expanded.contains("First"));
            assert!(expanded.contains("Second"));
            assert!(expanded.contains("Start"));
            assert!(expanded.contains("middle"));
            assert!(expanded.contains("end"));

            psh_free_result(result);
            psh_shutdown();
        }

        std::fs::remove_file(temp_file).ok();
    }
}
