local pl = (require 'pl.import_into')()

local targetN = 49

local targetdata = { delim = ',' }
targetdata['fieldnames'] = { "Filename", "Width", "Height" }

for i = 0, 9 do
  local filename = "0000"..tostring(i)..".ppm"
  targetdata[i] = {filename, 48, 48}
end

for i = 10, targetN do
  local filename = "000"..tostring(i)..".ppm"
  targetdata[i] = {filename, 48, 48}
end

local f = io.open("./FLRW.csv", 'w')
pl.data.new(targetdata):write(f)
io.close(f)