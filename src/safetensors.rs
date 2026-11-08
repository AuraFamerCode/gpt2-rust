//! Reader for the `.safetensors` format: an 8-byte little-endian header length,
//! a JSON header, then the raw tensor buffers. Everything is converted to `f32`.

use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub fn load<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Tensor>, String> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| format!("mmap {}: {e}", path.display()))?;
    let bytes: &[u8] = &mmap;

    if bytes.len() < 8 {
        return Err("file too short for a safetensors header".into());
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let body_start = 8 + header_len;
    if bytes.len() < body_start {
        return Err("truncated safetensors header".into());
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..body_start])
        .map_err(|e| format!("invalid safetensors header: {e}"))?;
    let obj = header
        .as_object()
        .ok_or("safetensors header is not a JSON object")?;
    let body = &bytes[body_start..];

    let mut tensors = HashMap::with_capacity(obj.len());
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info["dtype"]
            .as_str()
            .ok_or_else(|| format!("{name}: missing dtype"))?;
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .ok_or_else(|| format!("{name}: missing shape"))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let offs = info["data_offsets"]
            .as_array()
            .ok_or_else(|| format!("{name}: missing data_offsets"))?;
        let start = offs.first().and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let end = offs.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let raw = body
            .get(start..end)
            .ok_or_else(|| format!("{name}: data offsets out of range"))?;

        let data: Vec<f32> = match dtype {
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            other => return Err(format!("{name}: unsupported dtype {other}")),
        };

        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "{name}: expected {expected} elements, found {}",
                data.len()
            ));
        }
        tensors.insert(name.clone(), Tensor::new(shape, data));
    }
    Ok(tensors)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = (bits & 0x3ff) as f32;
    let val = match exp {
        0 => mant * 2f32.powi(-24),
        31 => {
            if mant == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        e => (1.0 + mant / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}
