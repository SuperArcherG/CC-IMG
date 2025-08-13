local args = { ... }
local filename = args[1]
local fps = tonumber(args[2]) or 20
local scale = tonumber(args[3]) or 0.5



if not filename then
    print("Usage: script_name <filename.bmp> or <folder> <fps> <scale>")
    return
end
if fps < 1 then
    delay = fps
else
    delay = 1 / fps 
end
local isDir = fs.isDir(filename)
local isFile = fs.exists(filename) and not isDir

if not isDir and not isFile then
    print("Error: '" .. filename .. "' does not exist.")
    return
end

local monitors = {
    {peripheral.wrap("monitor_8"), peripheral.wrap("monitor_9"), peripheral.wrap("monitor_10")},
    {peripheral.wrap("monitor_11"), peripheral.wrap("top"), peripheral.wrap("monitor_12")},
    {peripheral.wrap("monitor_13"), peripheral.wrap("monitor_14"), peripheral.wrap("monitor_16")}
}

local defaultMonText = {
    {"","",""},
    {"","",""},
    {"","",""}
}

local defaultTextColor = {
    {"","",""},
    {"","",""},
    {"","",""}
}


if not monitors or #monitors == 0 then error("No monitor(s) attached!") end

for row, monitorGroup in ipairs(monitors) do
    for column, monitor in ipairs(monitorGroup) do
        monitor.setTextScale(scale)
        monitor.setBackgroundColor(colors.black)
        monitor.clear()
    end
end

local w, h = monitors[1][1].getSize()
local sw, sh = monitors[1][1].getSize()

w = w * #monitors[1]
h = h * #monitors

print("Monitor resolution:", w, h)
print("Monitors:",#monitors[1], #monitors)

local hexMap = {}
for i = 0, 15 do
    hexMap[2^i] = string.format("%x", i)
end


local function calculateDefaults(frame)
    local frameHeight = #frame
    local frameWidth = #frame[1][1]
    for row, monitorGroup in ipairs(monitors) do -- Repeat for each row of monitors
        local offY = math.floor(sh * (row - 1)) -- Calculate the offset in pixels for the current row
        for column, monitor in ipairs(monitorGroup) do -- Repeat for each monitor in the row
            local offX = math.floor(sw * (column - 1)) -- Calculate the offset in pixels for the current monitor            
            local height = math.min(sh, #frame)
            local line = frame[1 + offY]
            if line then
                local bgColor = unpack(line)
                local sliceStart = offX + 1
                local sliceEnd = offX + (frameWidth - offX)
                local len = #(bgColor:sub(sliceStart, sliceEnd))
                defaultMonText[row + 1][column + 1] = string.rep(" ",len)
                defaultTextColor[row + 1][column + 1] = string.rep("0", len)
            end
        end
    end
end

local function renderFrame(frame)
    local frameHeight = #frame
    local frameWidth = #frame[1][1]

    for row, monitorGroup in ipairs(monitors) do -- Repeat for each row of monitors
        local offY = math.floor(sh * (row - 1)) -- Calculate the offset in pixels for the current row
        for column, monitor in ipairs(monitorGroup) do -- Repeat for each monitor in the row
            local offX = math.floor(sw * (column - 1)) -- Calculate the offset in pixels for the current monitor
            term.redirect(monitor) -- Redirect to the current monitor

            local height = math.min(sh, #frame)
            for y = 1, height do
                local line = frame[y + offY]
                if line then
                    local bgColor = unpack(line)

                    local sliceStart = offX + 1
                    local sliceEnd = offX + (frameWidth - offX)

                    if sliceEnd <= #bgColor then
                        term.setCursorPos(1, y)
                        term.blit(
                            defaultMonText[row + 1][column + 1],
                            defaultTextColor[row + 1][column + 1],
                            bgColor:sub(sliceStart, sliceEnd)
                        )
                    end
                end
            end
        end
    end
end

local function printPercent(label, pct)
    local x, y = term.getCursorPos()
    term.setCursorPos(1, y)         -- move to start of line
    term.clearLine()                -- wipe current line
    term.write(label .. ": " .. pct .. "%")
end

local function loadColorFrame(path)
    local file = fs.open(path, "r")
    local content = file.readAll()
    file.close()
    content = content:gsub("^return%s+", "")
    local ok, data = pcall(textutils.unserialize, content)
    if ok and type(data) == "table" and data[1] then
        return data
    end
end


local function base64_decode(data, yield_every)
    local b='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
    yield_every = yield_every or 32768

    -- Chunked gsub to avoid "too long without yielding"
    local cleaned = {}
    local chunkSize = 50000
    local pattern = '[^'..b..'=]'
    local total_chunks = math.ceil(#data / chunkSize)

    for chunk_i = 1, total_chunks do
        local start_i = (chunk_i - 1) * chunkSize + 1
        local chunk = data:sub(start_i, start_i + chunkSize - 1)
        chunk = chunk:gsub(pattern, '')
        cleaned[#cleaned+1] = chunk

        local pct = math.floor((chunk_i / total_chunks) * 100)
        printPercent("Base64 clean", pct)

        os.queueEvent("")
        os.pullEvent()
    end
    data = table.concat(cleaned)

    -- Decode loop
    local decoded = {}
    local c = 0
    local total = #data
    for i=1, total, 4 do
        local a = (b:find(data:sub(i,i), 1, true) or 1) - 1
        local b1 = (b:find(data:sub(i+1,i+1), 1, true) or 1) - 1
        local c1 = (b:find(data:sub(i+2,i+2), 1, true) or 1) - 1
        local d = (b:find(data:sub(i+3,i+3), 1, true) or 1) - 1
        local n = bit32.bor(
            bit32.lshift(a, 18),
            bit32.lshift(b1, 12),
            bit32.lshift(c1, 6),
            d
        )
        local x = string.char(
            bit32.band(bit32.rshift(n, 16), 255),
            bit32.band(bit32.rshift(n, 8), 255),
            bit32.band(n, 255)
        )
        decoded[#decoded+1] = x
        c = c + #x
        if c >= yield_every then
            local pct = math.floor((i / total) * 100)
            printPercent("Base64 decode", pct)
            os.queueEvent("")
            os.pullEvent()
            c = 0
        end
    end
    return table.concat(decoded)
end

-- Yield-safe zlib decompress using deflate.lua
local function decompress_zlib(data, yield_every)
    yield_every = yield_every or 32768
    local deflate = require("deflate")
    local output = {}
    local counter = 0
    local total = #data
    local processed = 0

    deflate.inflate_zlib{
        input = data,
        output = function(byte)
            output[#output+1] = string.char(byte)
            counter = counter + 1
            processed = processed + 1
            if counter >= yield_every then
                local pct = math.floor((processed / total) * 100)
                printPercent("Decompress", pct)
                os.queueEvent("")
                os.pullEvent()
                counter = 0
            end
        end
    }
    return table.concat(output)
end


local isCCAnim = false
if isFile and filename:match("%.ccanim$") then
    isCCAnim = true
end

if isDir then
    local frameFiles = fs.list(filename)
    table.sort(frameFiles, function(a, b)
        local aNum = tonumber(a:match("(%d+)")) or 0
        local bNum = tonumber(b:match("(%d+)")) or 0
        return aNum < bNum
    end)

    local frames = {}
    local frameIndex = 1
    local yieldCounter = 0

    for _, filename2 in ipairs(frameFiles) do
        if filename2:match("%.ccframe$") then
            local path = filename .. "/" .. filename2
            print("Loading frame:", path)
            local frame = loadColorFrame(path)
            if frame then
                table.insert(frames, frame)
            else
                print("Failed to load frame", path)
            end

            yieldCounter = yieldCounter + 1
            if yieldCounter >= 10 then
                os.queueEvent("")
                os.pullEvent()
                yieldCounter = 0
            end
        end
    end



    local numFrames = #frames
    if numFrames == 0 then
        error("No frames loaded!")
    end

    local totalTime = 0
    local renderedFrames = 0
    local calculatedDefaults = false
    while true do
        for _, frame in ipairs(frames) do
            local startTime = os.clock()
            if frame then
                if not calculatedDefaults then
                    calculateDefaults(frame)
                end
                renderFrame(frame)
            end
            frame = nil

            local frameTime = os.clock() - startTime
            if frameTime < delay then
                os.sleep(delay - frameTime)
            else
                os.queueEvent("")
                os.pullEvent()
            end

            local frameTime = os.clock() - startTime
            term.redirect(term.native())
            print(string.format("FPS = " .. math.floor((1/frameTime)*100+0.5)/100))

        end
    end
    -- local avg = totalTime / renderedFrames
    -- print(string.format("Average render time: %.4f seconds", avg))
elseif isCCAnim then
    local file = fs.open(filename, "r")
    local b64data = file.readAll()
    file.close()

    local b64content = b64data:match('return%s+"(.*)"')
    if not b64content then
        error("Invalid .ccanim format (missing return string)")
    end

    -- Decode base64 in chunks
    local compressed = base64_decode(b64content, 32768*2)
    -- Decompress ZLIB data with yields
    local lua_code = decompress_zlib(compressed, 32768*2)
    if not lua_code then
        error("Failed to decompress animation data")
    end

    -- Load Lua table
    local func, err = load(lua_code, filename, "t", {})
    if not func then
        error("Failed to load animation: " .. tostring(err))
    end
    local animTable = func()
    if type(animTable) ~= "table" or #animTable == 0 then
        error("Animation file did not return a valid table")
    end

    -- Playback
    while true do
        for _, frame in ipairs(animTable) do
            local startTime = os.clock()
            renderFrame(frame)
            local frameTime = os.clock() - startTime
            if frameTime < delay then
                os.sleep(delay - frameTime)
            else
                os.queueEvent("")
                os.pullEvent()
            end
        end
    end
else
    local frame = loadColorFrame(filename)
    renderFrame(frame)
end
term.redirect(term.native())

-- local totalEnd = os.clock()
-- print(string.format("Total script time: %.3f seconds", totalEnd - totalStart))

term.setBackgroundColor(colors.black)