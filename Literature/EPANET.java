	
package javaEPANET;


/*
**************************************************************************
**
**    Class  EPANET           
**
**************************************************************************
**    Copyright (C) 2010 M. E. Shafiee
**
**    This program is free software; you can redistribute it and/or modify
**    it under the terms of the GNU General Public License as published by
**    the Free Software Foundation; either version 2 of the License, or
**    (at your option) any later version.
**
**    This program is distributed in the hope that it will be useful,
**    but WITHOUT ANY WARRANTY; without even the implied warranty of
**    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**    GNU General Public License for more details.
**
**    You should have received a copy of the GNU General Public License
**    along with this program; if not, write to the Free Software
**    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
**************************************************************************
Please see ENgetcontrol method
*************************************************************************/
public class EPANET {

		private native void 	ENepanet(String f1, String f2, String f3);

		private native void 	ENopen(String f1, String f2, String f3);

		private native void 	ENclose();

		private native int		ENgetnodeindex(String id);

		private native String 	ENgetnodeid(int index);

		private native int 		ENgetnodetype(int index);
		
		private native float 	ENgetnodevalue(int index, int paramcode);

		private native int 		ENgetlinkindex(String id);

		private native String 	ENgetlinkid(int index);

		private native int 		ENgetlinktype(int ndex);

		private native int[] 	ENgetlinknodes(int index);

		private native float 	ENgetlinkvalue(int index, int paramcode);

		private native String 	ENgetpatternid(int index);

		private native int 		ENgetpatternindex(String id);

		private native int 		ENgetpatternlen(int index);

		private native float 	ENgetpatternvalue(int index, int period);

		//private native 			DATATransfer ENgetcontrol(int index); //this method doesn't work in JAVA EPANET Programmer's Toolkit

		private native int 		ENgetcount(int countcode);

		private native int 		ENgetflowunits();

		private native long 	ENgettimeparam(int paramcode);

		private native int 		ENgetqualtype(int qualcode);

		private native float 	ENgetoption(int optioncode);

		private native int 		ENgetversion();

		private native void 	ENsetcontrol(int cindex, int ctype, int lindex,
									float setting, int nindex, float level);

		private native void 	ENsetnodevalue(int index, int paramcode, float value);

		private native void 	ENsetlinkvalue(int index, int paramcode, float value);

		private native void 	ENsetpattern(int index, float[] factors, int nfactors);

		private native void 	ENsetpatternvalue(int index, int period, float value);

		private native void 	ENsetqualtype(int qualcode, String chemname,
									String chemunits, String tracenode);

		private native void 	ENsettimeparam(int paramcode, long timevalue);

		private native void 	ENsetoption(int optioncode, float value);

		private native void 	ENsavehydfile(String fname);

		private native void 	ENusehydfile(String fname);

		private native void 	ENsolveH();

		private native void 	ENopenH();

		private native void 	ENinitH(int flag);

		private native long 	ENrunH();

		private native long 	ENnextH();

		private native void 	ENcloseH();

		private native void 	ENsolveQ();

		private native void 	ENopenQ();

		private native void 	ENinitQ(int flag);

		private native long 	ENrunQ();

		private native long 	ENnextQ();

		private native long 	ENstepQ();

		private native void 	ENcloseQ();

		private native void 	ENsaveH();

		private native void 	ENsaveinpfile(String fname);

		private native void 	ENreport();

		private native void 	ENresetreport();

		private native void 	ENsetreport( String command );

		private native void 	ENsetstatusreport( int statuslevel );

		private native String 	ENgeterror( int errcode);
				
	
static {
	try {
		System.loadLibrary("Jepanet");
	} catch (UnsatisfiedLinkError e) {
		System.out.println("Error in Loading  Quality DLL");
	}
}
		


	// ----------------------------------------------------------------

		public void call_ENepanet(String input, String report, String output) {
			this.ENepanet(input, report, output);
			return;
		}
		
		public void call_ENopen(String input, String report, String output) {
			this.ENopen(input, report, output);
			return;
		}
		
		public void call_ENclose() {
			this.ENclose();
			return;
		}
		
		public int call_ENgetnodeindex(String id) {
			int returnval = this.ENgetnodeindex(id);
			return returnval;
		}
		
		public String call_ENgetnodeid(int index) {
			String returnval = this.ENgetnodeid(index);
			return returnval;
		}
		
		public int call_ENgetnodetype(int index) {
			int returnval = this.ENgetnodetype(index);
			return returnval;
		}
		
		public float call_ENgetnodevalue(int index, int paramcode) {
			float returnval = this.ENgetnodevalue(index, paramcode);
			return returnval;
		}
		
		public int call_ENgetlinkindex(String id) {
			int returnval = this.ENgetlinkindex(id);
			return returnval;
		}
		
		public String call_ENgetlinkid(int index) {
			String returnval = this.ENgetlinkid(index);
			return returnval;
		}
		
		public int call_ENgetlinktype(int index) {
			int returnval = this.ENgetlinktype(index);
			return returnval;
		}
		
		public int[] call_ENgetlinknodes(int index) {
			int[] returnval = this.ENgetlinknodes(index);
			return returnval;
		}
		
		public float call_ENgetlinkvalue(int index, int paramcode) {
			float returnval = this.ENgetlinkvalue(index, paramcode);
			return returnval;
		}

		public String call_ENgetpatternid(int index) {
			String returnval = this.ENgetpatternid(index);
			return returnval;
		}
		
		public int call_ENgetpatternindex(String id) {
			int returnval = this.ENgetpatternindex(id);
			return returnval;
		}
		
		public int call_ENgetpatternlen(int index) {
			int returnval = this.ENgetpatternlen(index);
			return returnval;
		}
		
		public float call_ENgetpatternvalue(int index, int period) {
			float returnval = this.ENgetpatternvalue(index, period);
			return returnval;
		}

		public int call_ENgetcount(int countcode) {
			int returnval = this.ENgetcount(countcode);
			return returnval;
		}
		
		public int call_ENgetflowunits() {
			int returnval = this.ENgetflowunits();
			return returnval;
		}	
		
		public long call_ENgettimeparam(int paramcode) {
			long returnval = this.ENgettimeparam(paramcode);
			return returnval;
		}
		
		public int call_ENgetqualtype(int qualcode) {
			int returnval = this.ENgetqualtype(qualcode);
			return returnval;
		}
		
		public float call_ENgetoption(int optioncode) {
			float returnval = this.ENgetoption(optioncode);
			return returnval;
		}
		
		public int call_ENgetversion() {
			int returnval = this.ENgetversion();
			return returnval;
		}	
		
		public void call_ENsetcontrol(int cindex, int ctype, int lindex,
				float setting, int nindex, float level) {
			this.ENsetcontrol(cindex, ctype, lindex, setting, nindex, level);
			return;
		}	
		
		public void call_ENsetnodevalue(int index, int paramcode, float value) {
			this.ENsetnodevalue(index, paramcode, value);
			return;
		}
		
		public void call_ENsetlinkvalue(int index, int paramcode, float value) {
			this.ENsetlinkvalue(index, paramcode, value);
			return;
		}
		
		public void call_ENsetpattern(int index, float[] factors, int nfactors) {
			this.ENsetpattern(index, factors, nfactors);
			return;
		}
		
		public void call_ENsetpatternvalue(int index, int period, float value) {
			this.ENsetpatternvalue(index, period, value);
			return;
		}
		
		public void call_ENsetqualtype(int qualcode, String chemname,
				String chemunits, String tracenode) {
			this.ENsetqualtype(qualcode, chemname, chemunits, tracenode);
			return;
		}	
		
		public void call_ENsettimeparam(int paramcode, long timevalue) {
			this.ENsettimeparam(paramcode, timevalue);
			return;
		}
		
		public void call_ENsetoption(int optioncode, float value) {
			this.ENsetoption(optioncode, value);
			return;
		}
		
		public void call_ENsavehydfile(String fname) {
			this.ENsavehydfile(fname);
			return;
		}
		
		public void call_ENusehydfile(String fname) {
			this.ENusehydfile(fname);
			return;
		}
		
		public void call_ENsolveH() {
			this.ENsolveH();
			return;
		}
		
		public void call_ENopenH() {
			this.ENopenH();
			return;
		}

		public void call_ENinitH(int flag) {
			this.ENinitH(flag);
			return;
		}

		public long call_ENrunH() {
			long returnval = this.ENrunH();
			return returnval;
		}

		public long call_ENnextH() {
			long returnval = this.ENnextH();
			return returnval;
		}

		public void call_ENcloseH() {
			this.ENcloseH();
			return;
		}

		public void call_ENsolveQ() {
			this.ENsolveQ();
			return;
		}
		
		public void call_ENopenQ() {
			this.ENopenQ();
			return;
		}
		
		public void call_ENinitQ(int flag) {
			this.ENinitQ(flag);
			return;
		}
		
		public long call_ENrunQ() {
			long returnval = this.ENrunQ();
			return returnval;
		}
		
		public long call_ENnextQ() {
			long returnval = this.ENnextQ();
			return returnval;
		}
		
		public long call_ENstepQ() {
			long returnval = this.ENstepQ();
			return returnval;
		}
		
		public void call_ENcloseQ() {
			this.ENcloseQ();
			return;
		}
		
		public void call_ENsaveH() {
			this.ENsaveH();
			return;
		}
		
		public void ENsaveinpfile() {
			this.ENsaveinpfile();
			return;
		}
		
		public void call_ENreport() {
			this.ENreport();
			return;
		}
		
		public void call_ENresetreport() {
			this.ENresetreport();
			return;
		}
		
		public void call_ENsetreport( String command ) {
			this.ENsetreport(command);
			return;
		}
			
		public void call_ENsetstatusreport( int statuslevel ) {
			this.ENsetstatusreport( statuslevel );
			return;
		}
			
		public String call_ENgeterror( int errcode) {
			String returnval = this.ENgeterror(errcode);
			return returnval;
		}



	private static final long serialVersionUID = 1L;

} 
