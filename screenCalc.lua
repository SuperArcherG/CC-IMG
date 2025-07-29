-- Configuration
local monitorSide = "right" -- Change if needed
local filePath = "SIZES.txt"
local textScale = 0.5
local sleepTime = 0.5 -- seconds between checks

-- Helper to serialize size
local function sizeToString(w, h)
    return w .. "x" .. h
end

-- Helper to read all known sizes from file
local function loadSizes()
    local sizes = {}
    if fs.exists(filePath) then
        local file = fs.open(filePath, "r")
        for line in file.readLine do
            sizes[line] = true
        end
        file.close()
    end
    return sizes
end

-- Initialize monitor
local monitor = peripheral.wrap(monitorSide)
if not monitor then
    error("No monitor found on side: " .. monitorSide)
end

monitor.setTextScale(textScale)

local knownSizes = loadSizes()
local lastSize = ""

while true do
    local w, h = monitor.getSize()
    local currentSize = sizeToString(w, h)

    if currentSize ~= lastSize and not knownSizes[currentSize] then
        -- Log new size
        local file = fs.open(filePath, "a")
        file.writeLine(currentSize)
        file.close()

        print("New monitor size detected: " .. currentSize)
        knownSizes[currentSize] = true
        lastSize = currentSize
    end

    sleep(sleepTime)
end
