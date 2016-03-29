----------------------------------------------------------
-- conditional mixture of factor analyser
----------------------------------------------------------

local VBCMFA = require 'VBCMFA'

local CFMA = {}

local parent = VBCMFA:_factory()
setmetatable(CFMA, parent)
parent.__index = parent


function CFMA:new(...)
   local o = {}
   setmetatable(o, self)
   self.__index = self
   o:__init(...)
   return o
end


function CFMA:__init(cfg)
   self.debug = cfg.debug
   self.pause = cfg.pause
   parent:__init(cfg)
end


-- Mu  means of factor analyzers of intent
-- Pti responsibilities
function CFMA:train(Mu, Pti, epochs)
   local n, s, p, k, d, N = self:_setandgetDims()

   for epoch = 1, epochs do
      for subEpoch = 1, 10 do
         self:infer("Qz", Pt, Y)
         self:infer("HyperX", pause)
         self:infer("Qx", Pt, Y)
      end

      for subEpoch = 1, 10 do
         self:infer("QL", Pt, Y)
         self:infer("QG", Pt, Y)
      end

      for subEpoch = 1, 10 do
         self:infer("Qnu")
         self:infer("Qomega")
      end

      for subEpoch = 1, 10 do
         self:infer("Qpi")
         self:infer("Qs", Pt, Y)
         t = os.clock()
         self:inferQs(Pt, Y)
         print(string.format("Qs\t= %f", os.clock() - t))
      end

      for subEpoch = 1, 10 do
         self:infer("PsiI", Pt, Y)
      end

      if self.debug then
         self:print("Zm", "hidden")
         self:print("Zcov", "hidden")
         self:print("Xm", "conditional")
         self:print("Xcov", "conditional")
         self:print("Lm", "factorLoading")
         self:print("Lcov", "factorLoading")
         self:print("Gm", "factorLoading")
         self:print("Gcov", "factorLoading")
      end

      collectgarbage()
   end
end


function CFMA:infer(tr, ...)
   local t = os.clock()
   print(string.format("\n===== Infer%s =====\n", tr))
   self["infer"..tr](self, ...)
   print(string.format("\tt for %s = %f", tr, os.clock() - t))
   print("--------------------------------------------------")
   if self.pause == 1 then io.read() end
end


function CFMA:print(tr, ns)
   print(string.format("%s after inference = \n", tr))
   print(self[ns][tr])
   if self.pause == 1 then io.read() end
end


return CFMA