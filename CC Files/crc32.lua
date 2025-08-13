-- crc32.lua — IEEE CRC-32 (poly 0xEDB88320) for ComputerCraft (Lua 5.1 + bit32)

local bit = bit32
local band, bxor, rshift = bit.band, bit.bxor, bit.rshift

-- Build CRC lookup table
local function make_table()
    local t = {}
    for i = 0, 255 do
        local crc = i
        for _ = 1, 8 do
            if band(crc, 1) ~= 0 then
                crc = bxor(rshift(crc, 1), 0xEDB88320)
            else
                crc = rshift(crc, 1)
            end
        end
        t[i] = crc
    end
    return t
end

local CRC_TABLE = make_table()

local M = {}

-- Compute CRC-32 of a string
function M.crc32(str, crc)
    crc = crc or 0xFFFFFFFF
    for i = 1, #str do
        local b = string.byte(str, i)
        local idx = band(bxor(crc, b), 0xFF)
        crc = bxor(rshift(crc, 8), CRC_TABLE[idx])
    end
    return bxor(crc, 0xFFFFFFFF)
end

return M
