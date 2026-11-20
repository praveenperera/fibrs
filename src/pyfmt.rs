//! Python and NumPy semantics the reference output depends on
//!
//! The reference's numbers pass through Numba's `min`/`max`, NumPy's unstable argsort, NumPy's pairwise
//! summation and Python's `json.dumps`; each is reproduced here exactly, so the native output can be
//! identical rather than merely close

use serde_json::Value;

/// Numba's (and Python's) two-argument `max`: the second argument wins only when strictly greater
#[inline(always)]
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Numba's (and Python's) two-argument `min`: the second argument wins only when strictly smaller
#[inline(always)]
pub fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// `np.argsort(values)` with the default kind for a 16-bit key: NumPy's introsort (`aquicksort_`), whose
/// order among equal keys this reproduces
pub fn argsort_u16(values: &[u16]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    aquicksort(values, &mut idx);
    idx
}

// numpy 2.4.6 npysort_common.h
const SMALL_QUICKSORT: usize = 15;

/// Partitions at least this long that the quicksort sets aside are sorted on the pool
const PARALLEL_PARTITION: usize = 1 << 15;

/// NumPy's `aquicksort_` with the partitions it pushes on its stack sorted on the pool
///
/// A pushed partition is sorted the same way whenever it is popped: it depends only on its own elements and the
/// depth limit pushed with it, and partitions are disjoint. So sorting the larger ones on other threads gives
/// NumPy's order exactly, equal keys included
fn aquicksort(v: &[u16], tosort: &mut [usize]) {
    let num = tosort.len();
    if num < 2 {
        return;
    }
    let cdepth = 2 * msb(num) as i64;
    rayon::in_place_scope(|scope| sort_popped(v, tosort, cdepth, scope));
}

/// One range as NumPy's loop takes it off the stack: heapsort past the depth limit, otherwise partition and keep
/// the smaller side until it is small, then insertion sort
fn sort_popped<'a>(v: &'a [u16], part: &'a mut [usize], cdepth: i64, scope: &rayon::Scope<'a>) {
    let mut stack: Vec<(&'a mut [usize], i64)> = vec![(part, cdepth)];
    while let Some((part, mut cdepth)) = stack.pop() {
        if cdepth < 0 {
            aheapsort(v, part);
            continue;
        }
        let mut part = part;
        while part.len() - 1 > SMALL_QUICKSORT {
            let pi = partition(v, part);
            cdepth -= 1;
            let (left, rest) = std::mem::take(&mut part).split_at_mut(pi);
            let right = &mut rest[1..];
            // push the larger partition, keep sorting the smaller one
            let (kept, pushed) = if left.len() < right.len() { (left, right) } else { (right, left) };
            part = kept;
            if pushed.len() >= PARALLEL_PARTITION {
                scope.spawn(move |scope| sort_popped(v, pushed, cdepth, scope));
            } else {
                stack.push((pushed, cdepth));
            }
        }
        insertion_sort(v, part);
    }
}

/// Median-of-three partition of `part` as `aquicksort_` does it (pl = 0, pr = the last index); returns the pivot's
/// final index
fn partition(v: &[u16], tosort: &mut [usize]) -> usize {
    let less = |a: usize, b: usize| v[a] < v[b];
    let (pl, pr) = (0, tosort.len() - 1);
    let pm = pl + ((pr - pl) >> 1);
    if less(tosort[pm], tosort[pl]) {
        tosort.swap(pm, pl);
    }
    if less(tosort[pr], tosort[pm]) {
        tosort.swap(pr, pm);
    }
    if less(tosort[pm], tosort[pl]) {
        tosort.swap(pm, pl);
    }
    let vp = v[tosort[pm]];
    let mut pi = pl;
    let mut pj = pr - 1;
    tosort.swap(pm, pj);
    loop {
        loop {
            pi += 1;
            if v[tosort[pi]] >= vp {
                break;
            }
        }
        loop {
            pj -= 1;
            if vp >= v[tosort[pj]] {
                break;
            }
        }
        if pi >= pj {
            break;
        }
        tosort.swap(pi, pj);
    }
    tosort.swap(pi, pr - 1);
    pi
}

fn insertion_sort(v: &[u16], tosort: &mut [usize]) {
    for i in 1..tosort.len() {
        let vi = tosort[i];
        let vp = v[vi];
        let mut j = i;
        while j > 0 && vp < v[tosort[j - 1]] {
            tosort[j] = tosort[j - 1];
            j -= 1;
        }
        tosort[j] = vi;
    }
}

fn msb(mut n: usize) -> u32 {
    let mut depth = 0;
    while {
        n >>= 1;
        n != 0
    } {
        depth += 1;
    }
    depth
}

fn aheapsort(v: &[u16], tosort: &mut [usize]) {
    let n0 = tosort.len();
    // one-based heap indexing, as in NumPy
    let a = |i: usize| i - 1;
    let less = |x: usize, y: usize| v[x] < v[y];
    let mut l = n0 >> 1;
    while l > 0 {
        let tmp = tosort[a(l)];
        let mut i = l;
        let mut j = l << 1;
        while j <= n0 {
            if j < n0 && less(tosort[a(j)], tosort[a(j + 1)]) {
                j += 1;
            }
            if less(tmp, tosort[a(j)]) {
                tosort[a(i)] = tosort[a(j)];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[a(i)] = tmp;
        l -= 1;
    }
    let mut n = n0;
    while n > 1 {
        let tmp = tosort[a(n)];
        tosort[a(n)] = tosort[a(1)];
        n -= 1;
        let mut i = 1;
        let mut j = 2;
        while j <= n {
            if j < n && less(tosort[a(j)], tosort[a(j + 1)]) {
                j += 1;
            }
            if less(tmp, tosort[a(j)]) {
                tosort[a(i)] = tosort[a(j)];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[a(i)] = tmp;
    }
}

/// `np.add.reduce` of a contiguous float64 vector: the identity 0.0 plus NumPy's pairwise sum
pub fn np_sum_f64(a: &[f64]) -> f64 {
    0.0 + pairwise(a)
}

fn pairwise(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0;
        for &x in a {
            res += x;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise(&a[..n2]) + pairwise(&a[n2..])
}

/// Python's `repr` of a finite float (shortest round-trip digits, Python's switch to exponent form)
pub fn py_float_repr(x: f64) -> String {
    assert!(x.is_finite(), "JSON cannot hold a non-finite float");
    // Rust's shortest form gives the digit count; Python's dtoa breaks an exact tie at that count to
    // even, while the shortest formatter rounds it up, so the digits come from the correctly rounded
    // fixed-precision form whenever that still round-trips
    let shortest = format!("{:e}", x.abs());
    let count = shortest.split_once('e').map_or(0, |(m, _)| m.chars().filter(char::is_ascii_digit).count());
    let rounded = format!("{:.*e}", count.saturating_sub(1), x.abs());
    let sci = if rounded.parse::<f64>().is_ok_and(|v| v == x.abs()) { rounded } else { shortest };
    let (mantissa, exp) = sci.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let digits = if digits.chars().all(|c| c == '0') { "0".to_string() } else { digits };
    let sign = if x.is_sign_negative() { "-" } else { "" };
    let decpt = exp + 1;
    if x != 0.0 && (decpt <= -4 || decpt > 16) {
        let (head, tail) = digits.split_at(1);
        let frac = if tail.is_empty() { String::new() } else { format!(".{tail}") };
        let esign = if exp < 0 { '-' } else { '+' };
        return format!("{sign}{head}{frac}e{esign}{:02}", exp.abs());
    }
    let decpt = if x == 0.0 { 1 } else { decpt };
    let n = digits.len() as i32;
    let body = if decpt <= 0 {
        format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
    } else if decpt < n {
        format!("{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
    } else {
        format!("{}{}.0", digits, "0".repeat((decpt - n) as usize))
    };
    format!("{sign}{body}")
}

/// How Python's `json.dumps` lays a value out: its `indent`, `separators` and `sort_keys`
#[derive(Clone, Copy, Debug)]
pub struct JsonStyle {
    /// Spaces per nesting level; `None` writes everything on one line
    pub indent: Option<usize>,
    /// Between items, and between a key and its value
    pub item: &'static str,
    pub key: &'static str,
    pub sort_keys: bool,
}

impl JsonStyle {
    /// `json.dumps(value)`: one line, `", "` and `": "`
    pub const DEFAULT: Self = Self { indent: None, item: ", ", key: ": ", sort_keys: false };
    /// `separators=(",", ":")`
    pub const COMPACT: Self = Self { indent: None, item: ",", key: ":", sort_keys: false };

    /// `json.dumps(value, indent=n)`, whose item separator drops its trailing space
    pub const fn indented(n: usize) -> Self {
        Self { indent: Some(n), item: ",", key: ": ", sort_keys: false }
    }

    pub const fn sorted(self) -> Self {
        Self { sort_keys: true, ..self }
    }
}

/// `json.dumps(value, ...)` in `style`, with Python's default `ensure_ascii` and `allow_nan=False`
pub fn py_dumps(value: &Value, style: JsonStyle) -> String {
    let mut out = String::new();
    write_value(value, style, 0, &mut out);
    out
}

/// `json.dumps(value, separators=(",", ":"), allow_nan=False)` with Python's default `ensure_ascii`
pub fn py_json(value: &Value) -> String {
    py_dumps(value, JsonStyle::COMPACT)
}

/// `json.dump(value, handle, indent=2)`: one item per line, `": "` after keys, empty containers inline
pub fn py_json_indent(value: &Value) -> String {
    py_dumps(value, JsonStyle::indented(2))
}

fn write_value(value: &Value, style: JsonStyle, depth: usize, out: &mut String) {
    // a newline and the next level's indent before each item, or the item separator alone on one line
    let pad = |out: &mut String, d: usize| {
        if let Some(n) = style.indent {
            out.push('\n');
            out.push_str(&" ".repeat(n * d));
        }
    };
    match value {
        Value::Array(items) if !items.is_empty() => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(style.item);
                }
                pad(out, depth + 1);
                write_value(item, style, depth + 1, out);
            }
            pad(out, depth);
            out.push(']');
        }
        Value::Object(map) if !map.is_empty() => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if style.sort_keys {
                // Python sorts str keys by code point, which is the UTF-8 byte order
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (i, (k, v)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(style.item);
                }
                pad(out, depth + 1);
                write_str(k, out);
                out.push_str(style.key);
                write_value(v, style, depth + 1, out);
            }
            pad(out, depth);
            out.push('}');
        }
        Value::Array(_) => out.push_str("[]"),
        Value::Object(_) => out.push_str("{}"),
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&py_float_repr(n.as_f64().expect("number")));
            }
        }
        Value::String(s) => write_str(s, out),
    }
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

#[cfg(test)]
pub(crate) mod tests {
    /// NumPy's `aquicksort_` as it was ported, on one thread: the reference for the pooled [`argsort_u16`]
    pub(crate) fn serial_argsort_u16(values: &[u16]) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..values.len()).collect();
        serial_aquicksort(values, &mut idx);
        idx
    }

    fn serial_aquicksort(v: &[u16], tosort: &mut [usize]) {
        let num = tosort.len();
        if num < 2 {
            return;
        }
        let less = |a: usize, b: usize| v[a] < v[b];
        // indices into `tosort`; pr is inclusive, as in npysort/quicksort.cpp
        let mut pl: usize = 0;
        let mut pr: usize = num - 1;
        let mut stack: Vec<(usize, usize, i64)> = Vec::new();
        let mut cdepth: i64 = 2 * msb(num) as i64;
        loop {
            let mut heap = false;
            if cdepth < 0 {
                aheapsort(v, &mut tosort[pl..=pr]);
                heap = true;
            } else {
                while pr - pl > SMALL_QUICKSORT {
                    let pm = pl + ((pr - pl) >> 1);
                    if less(tosort[pm], tosort[pl]) {
                        tosort.swap(pm, pl);
                    }
                    if less(tosort[pr], tosort[pm]) {
                        tosort.swap(pr, pm);
                    }
                    if less(tosort[pm], tosort[pl]) {
                        tosort.swap(pm, pl);
                    }
                    let vp = v[tosort[pm]];
                    let mut pi = pl;
                    let mut pj = pr - 1;
                    tosort.swap(pm, pj);
                    loop {
                        loop {
                            pi += 1;
                            if v[tosort[pi]] >= vp {
                                break;
                            }
                        }
                        loop {
                            pj -= 1;
                            if vp >= v[tosort[pj]] {
                                break;
                            }
                        }
                        if pi >= pj {
                            break;
                        }
                        tosort.swap(pi, pj);
                    }
                    let pk = pr - 1;
                    tosort.swap(pi, pk);
                    // push the larger partition, keep sorting the smaller one
                    if pi - pl < pr - pi {
                        cdepth -= 1;
                        stack.push((pi + 1, pr, cdepth));
                        pr = pi - 1;
                    } else {
                        cdepth -= 1;
                        stack.push((pl, pi - 1, cdepth));
                        pl = pi + 1;
                    }
                }
            }
            if !heap {
                for i in pl + 1..=pr {
                    let vi = tosort[i];
                    let vp = v[vi];
                    let mut j = i;
                    while j > pl && vp < v[tosort[j - 1]] {
                        tosort[j] = tosort[j - 1];
                        j -= 1;
                    }
                    tosort[j] = vi;
                }
            }
            match stack.pop() {
                Some((l, r, d)) => {
                    pl = l;
                    pr = r;
                    cdepth = d;
                }
                None => break,
            }
        }
    }

    use super::*;
    use serde_json::json;

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (9.362, "9.362"),
            (9.362 / 1000.0, "0.009362"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (1e16, "1e+16"),
            (1234567890123456.0, "1234567890123456.0"),
            (12345.678, "12345.678"),
            (0.1 + 0.2, "0.30000000000000004"),
            (-24096.5, "-24096.5"),
            // exactly 12540.5968017578125, a tie at 17 digits, which Python rounds to even
            (12540.596801757812, "12540.596801757812"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
        ];
        for (x, want) in cases {
            assert_eq!(py_float_repr(x), want, "{x:e}");
        }
    }

    #[test]
    fn json_matches_python_dumps() {
        let v = json!({"a": [1, 2.5, "x\"y"], "b": {"c": false, "d": null}, "é": 0.1});
        assert_eq!(py_json(&v), r#"{"a":[1,2.5,"x\"y"],"b":{"c":false,"d":null},"\u00e9":0.1}"#);
    }

    #[test]
    fn indented_json_matches_python_dump() {
        let v = json!({"a": [1, [], {}], "b": {"c": 9.362}, "d": null});
        let want = "{\n  \"a\": [\n    1,\n    [],\n    {}\n  ],\n  \"b\": {\n    \"c\": 9.362\n  },\n  \"d\": null\n}";
        assert_eq!(py_json_indent(&v), want);
    }

    #[test]
    fn styles_match_python_dumps() {
        let v = json!({"b": [1, 0.0, []], "a": {"é": null}});
        assert_eq!(py_dumps(&v, JsonStyle::DEFAULT), r#"{"b": [1, 0.0, []], "a": {"\u00e9": null}}"#);
        assert_eq!(py_dumps(&v, JsonStyle::DEFAULT.sorted()), r#"{"a": {"\u00e9": null}, "b": [1, 0.0, []]}"#);
        let want = "{\n    \"a\": {\n        \"\\u00e9\": null\n    },\n    \"b\": [\n        1,\n        0.0,\n        []\n    ]\n}";
        assert_eq!(py_dumps(&v, JsonStyle::indented(4).sorted()), want);
    }

    #[test]
    fn argsort_sorts_and_is_a_permutation() {
        let mut state = 12345u64;
        let values: Vec<u16> = (0..5000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) % 300) as u16
            })
            .collect();
        let order = argsort_u16(&values);
        assert!(order.windows(2).all(|w| values[w[0]] <= values[w[1]]));
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..values.len()).collect::<Vec<_>>());
    }

    /// Sorting the larger partitions on the pool gives NumPy's order exactly, among equal keys too
    #[test]
    fn pooled_argsort_is_the_serial_order() {
        let mut state = 7u64;
        let values: Vec<u16> = (0..1_500_000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) % 40) as u16
            })
            .collect();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        assert!(pool.install(|| argsort_u16(&values)) == serial_argsort_u16(&values));
    }

    #[test]
    fn pairwise_sum_blocks() {
        let a: Vec<f64> = (0..1000).map(|i| (i as f64).sqrt() * 0.1).collect();
        let naive: f64 = a.iter().sum();
        assert!((np_sum_f64(&a) - naive).abs() < 1e-9);
    }
}
