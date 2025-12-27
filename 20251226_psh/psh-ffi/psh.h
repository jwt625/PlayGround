/*
 * psh FFI - C Header for Swift Integration
 *
 * This header provides C-compatible bindings for the psh-core Rust library.
 * Use this header to integrate psh into Swift via a bridging header.
 */

#ifndef PSH_H
#define PSH_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Type Definitions */

/**
 * C-compatible string (null-terminated char pointer)
 */
typedef char* PshCString;

/**
 * Warning structure
 *
 * warning_type values:
 *   0 = UnknownNamespace
 *   1 = UnknownOperation
 *   2 = UnknownKey
 */
typedef struct {
    uint32_t warning_type;
    PshCString namespace;
    PshCString name;
} PshWarning;

/**
 * Expansion result structure
 *
 * Either `text` is set (success) or `error` is set (failure).
 * Always check `error` first - if it's NULL, the expansion succeeded.
 */
typedef struct {
    PshCString text;
    PshWarning* warnings;
    size_t warning_count;
    PshCString error;
} PshExpandResult;

/* Function Declarations */

/**
 * Initialize the psh engine with snippets from a TOML file
 *
 * @param snippets_path Path to the snippets.toml file (null-terminated)
 * @return true if initialization succeeded, false otherwise
 *
 * This must be called before any other psh functions.
 * Can be called multiple times to reload snippets.
 */
bool psh_init(const char* snippets_path);

/**
 * Expand text containing psh directives
 *
 * @param text Input text with psh directives (null-terminated)
 * @return Pointer to PshExpandResult (must be freed with psh_free_result)
 *
 * The returned result contains either:
 * - Expanded text and warnings (if successful)
 * - Error message (if failed)
 *
 * Always free the result with psh_free_result when done.
 */
PshExpandResult* psh_expand(const char* text);

/**
 * Free a PshExpandResult returned by psh_expand
 *
 * @param result Pointer to result to free
 *
 * This frees all memory associated with the result, including:
 * - The text string
 * - The error string
 * - All warning structures
 * - The result structure itself
 */
void psh_free_result(PshExpandResult* result);

/**
 * Reload snippets from a file
 *
 * @param snippets_path Path to the snippets.toml file (null-terminated)
 * @return true if reload succeeded, false otherwise
 *
 * This is equivalent to calling psh_init again.
 */
bool psh_reload_snippets(const char* snippets_path);

/**
 * Shutdown the psh engine and free all resources
 *
 * After calling this, you must call psh_init again before using psh.
 */
void psh_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* PSH_H */

