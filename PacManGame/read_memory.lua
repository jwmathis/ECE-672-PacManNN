-- Set the correct memory domain
memory.usememorydomain("IWRAM")

-- Define the address
local address = 0x7268

-- File path to store pellet count (adjust this path based on your setup)
local file_path = "pellet_count.txt"

-- Frequency of writing to file (in frames)
local write_interval = 100
local frame_count = 0

-- Function to write pellet count to a file
function write_pellet_count_to_file(count)
    local file = io.open(file_path, "w")
    if file then
        file:write(count)
        file:close()
    else
        console.log("Error: Unable to open file for writing")
    end
end

-- Main loop that runs every frame
while true do
    -- Read the value at the pellet address
    local pellet_count = memory.read_u32_le(address)
    
    -- Print the pellet count to BizHawk console (for debugging)
    --console.log("Pellet Count: " .. pellet_count)
    
    -- Write to file every `write_interval` frames
    frame_count = frame_count + 1
    if frame_count >= write_interval then
        write_pellet_count_to_file(pellet_count)
        frame_count = 0
    end
    
    -- Wait until the next frame
    emu.frameadvance()
end
