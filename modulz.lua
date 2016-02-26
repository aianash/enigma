-----------------------------------------------------
--[[ modulz ]]--
-- Utility function to assist OOP and lazy submodules
-- inside a primary package
-- Usage:
-- (require 'modulz')('primary pkg name')
-----------------------------------------------------

local pl = (require 'pl.import_into')()
pl.stringx.import()

-- create a loading module which
-- with overloaded __index function
-- looks for a file and requires it
local function lazyloadingmodule(name)
   local mod = { relmodname = '', submodules = {} } -- lazy loaded

   function mod:_relativemodname(name)
      if self.relmodname ~= '' then
         return self.relmodname.."."..name
      else 
         return name
      end
   end

   if name then mod.relmodname = name end

   local modmt = {}

   -- override __index to lazy load
   -- any missing modules
   function modmt.__index(t, name)
      local mod = require(t:_relativemodname(name))
      if mod then
         rawset(t, name, mod)
         return t[name]
      else
         error(string.format('No module named %s present', name))
      end
   end

   setmetatable(mod, modmt)

   return mod
end

--
local function getOrCreateSubmodule(name, parent)
   assert(type(parent) == 'table', 'Parent is not yet created')
   if not parent.submodules[name] then
      parent[name] = lazyloadingmodule(parent:_relativemodname(name))
      parent.submodules[name] = true
      return parent[name]
   else
      return parent[name]
   end
end

--
local function constructortbl(metatable)
   local ct = {}
   setmetatable(ct, {
      __index=metatable,
      __newindex=metatable,
      __metatable=metatable,
      __call=function(self, ...)
         return self.new(...)
      end
   })
   return ct
end

--
return function (primarypkgname)

   local function withinprimary(name) return name:lfind(primarypkgname, 1) ~= nil end   

   _G[primarypkgname] = lazyloadingmodule()

   -- create a submodule where, automatically create
   -- parent if not present
   _G.submodule = function (name)
      assert(withinprimary(name), 'Submodule not inside primary package '..primarypkgname)

      local mod = name:split('.', 2)[2]
      assert(mod, 'No name for submodules')

      local parent = _G[primarypkgname]

      for _, submod in ipairs(mod:split('.')) do
         parent = getOrCreateSubmodule(submod, parent)
      end
      return parent
   end

   --
   _G.import = function (name)
      assert(withinprimary(name), 'Submodule not inside primary package '..primarypkgname)

      local mod = name:split('.', 2)[2]
      assert(mod, 'No name for submodules')

      local parent = _G[primarypkgname]

      for _, submod in ipairs(mod:split('.')) do
         parent = parent[submod]
      end

      return parent
   end

   --
   _G.klazz = function (name, parentname)
      assert(type(name) == 'string', "Class's name should be string")
      assert(withinprimary(name), 'Both class and parent class should be with primary pkg = '..primarypkgname) 
      
      local mt = { __typename = name }

      mt.__index = mt

      mt.__factory =
         function ()
            local self = {}
            setmetatable(self, mt)
            return self
         end

      mt.__init =
         function () end

      mt.new =
         function(...)
            local self = mt.__factory()
            self:__init(...)
            return self
         end

      if not parentname then
         return constructortbl(mt)
      elseif withinprimary(parentname) then
         local parent = import(parentname)
         setmetatable(mt, getmetatable(parent))
         return constructortbl(mt), parent
      else
         error('Parent name present but not within primary package')
      end
   end

   return _G[primarypkgname]

end