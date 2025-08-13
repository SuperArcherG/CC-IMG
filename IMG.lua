---@diagnostic disable: undefined-field, deprecated
local args = { ... }
local filename = args[1]
local fps = tonumber(args[2]) or 20
local scale = tonumber(args[3]) or 0.5



if not filename then
    print("Usage: script_name <filename.bmp> or <folder> <fps> <scale>")
    return
end
if fps < 1 then
    Delay = fps
else
    Delay = 1 / fps 
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
            printPercent("Loading frame", path)
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
            if frameTime < Delay then
                os.sleep(Delay - frameTime)
            else
                os.queueEvent("")
                os.pullEvent()
            end

            local frameTime = os.clock() - startTime
            term.redirect(term.native())
            printPercent("FPS", math.floor((1/frameTime)*100+0.5)/100)

        end
    end
    -- local avg = totalTime / renderedFrames
    -- print(string.format("Average render time: %.4f seconds", avg))
else
    local frame = loadColorFrame(filename)
    renderFrame(frame)
end
term.redirect(term.native())

-- local totalEnd = os.clock()
-- print(string.format("Total script time: %.3f seconds", totalEnd - totalStart))

term.setBackgroundColor(colors.black)