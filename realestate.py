from typing import Annotated, Optional, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pdfrw import PdfReader, PdfWriter, PdfName
from typing import Dict, List
import os
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import smtplib

SYSTEM_PROMPT = """You are a real estate document assistant specializing in preparing offer documents.
Your responsibilities include:

1. Extracting key information from realtor messages including:
   - Buyer and seller names
   - Property details
   - Offer amounts
   - Conditions
   - Important dates

2. Accurately filling out standard real estate forms including:
   - Purchase agreements
   - Deposit receipts
   - Condition forms
   - All supporting documentation

3. Ensuring all required fields are completed with proper formatting:
   - Currency values should include dollar signs and commas
   - Dates should be in YYYY-MM-DD format
   - Names should be in proper case
   - Addresses should be complete and properly formatted

4. Validating that all required information is present before document preparation

If any required information is missing or unclear, ask for clarification before proceeding.
Never make assumptions about missing information."""

# Create a PDF handler class to fill PDF forms
class PDFHandler:
    def __init__(self, template_dir: str):
        self.template_dir = template_dir
     
    def fill_pdf_form(self, template_name: str, output_path: str, filed_data: Dict) -> str:
        """Fill the PDF form with the given data and save it to the output path."""
        template_path = os.path.join(self.template_dir, template_name)
        template_pdf = PdfReader(template_path)
        
        for page in template_pdf.pages:
            annotations = page['/Annots']
            if annotations:
                for annotation in annotations:
                    key = annotation.get('/T').to_unicode()
                    if key in filed_data:
                        annotation.update({
                            PdfName('V'): str(filed_data[key]),
                            PdfName('Ff'): 1
                        })
        
        PdfWriter().write(output_path, template_pdf)
        return output_path


class OfferDetails(BaseModel)                     :
    """Structure for the offer details."""
    buyer_name: str = Field(..., title="Buyer Name", description="Name of the buyer.")
    seller_name: str = Field(..., title="Seller Name", description="Name of the seller.")
    property_address: str = Field(..., title="Property Address", description="Address of the property.")
    offer_price: float = Field(..., title="Offer Price", description="Price of the offer.")
    deposit: float = Field(..., title="Deposit", description="Deposit amount.")
    conditions: List[str] = Field(..., title="Conditions", description="List of conditions.")
    closing_date: str = Field(..., title="Closing Date", description="Closing date of the offer.")
    
class State(TypedDict):
    """Workflow state structure."""
    messages: Annotated[list, add_messages]
    offer_details: Optional[OfferDetails]
    competed_pdfs: List[str]
    errors: Optional[str]
    
# Configuration
PDF_TEMPLATES = {
    "purchase_agreement": "purchase_agreement.pdf",
    "deposit_receipt": "deposit_receipt.pdf",
    "conditions_form": "conditions_form.pdf",
}
# Define the directory paths for templates and output
TEMPLATE_DIR = "/path/to/pdf/templates"
OUTPUT_DIR = "/path/to/output/pdfs"

class OfferAgent:
    def __init__(self):
        self.pdf_handler = PDFHandler(TEMPLATE_DIR)
        self.llm = ChatAnthropic(temperature=0)
        
    def parse_offer_details(self, message: str) -> OfferDetails:
        """Parse offer details from the message using LLm"""
        prompt = f"""Extract the following information from the message:
        - Buyer's name
        - Seller's name
        - Property address
        - Offer price
        - Deposit amount
        - List of conditions
        
        Message: {message}
        
        Return the information in a structured format."""
        
        response = self.llm.invoke(prompt)
        # Parse the response to extract the offer details
        # This is a simplified version, you can use more advanced NLP techniques to extract the information
        return OfferDetails(
            buyer_name="John Doe",
            seller_name="Alice Smith",
            property_address="123 Main St, New York, NY",
            offer_price=500000,
            deposit=50000,
            conditions=["Subject to inspection", "Subject to financing"],
            closing_date="2022-12-31"
        )
     
    def validate_offer_details(self, offer_details: OfferDetails) -> List[str]:
        """Validate the offer details for completeness and correctness."""
        errors = []
        
        if not offer_details.buyer_name or len(offer_details.buyer_name) < 2:
            errors.append("Buyer name is missing or too short.")
            
        if not offer_details.seller_name or len(offer_details.seller_name) < 2:
            errors.append("Seller name is missing or too short.")
            
        if not offer_details.property_address or len(offer_details.property_address) < 5:
            errors.append("Property address is missing or too short.")
            
        if not offer_details.offer_price or offer_details.offer_price <= 0:
            errors.append("Offer price is missing or invalid.")
            
        return errors
        
    def prepare_documents(self, state: State):
        """Prepare all offer documents"""
        try:
            offer_details = self.parse_offer_details(state["messages"][-1].content)
            completed_pdfs = []
            
            for template_name, doc_type in PDF_TEMPLATES.items():
                output_path = os.path.join(OUTPUT_DIR, f"{doc_type}_{state['offer_details'].buyer_name}.pdf")
                
                
                # Map offer details to the PDF fields
                field_data = {
                    "Buyer Name": offer_details.buyer_name,
                    "Seller Name": offer_details.seller_name,
                    "Property Address": offer_details.property_address,
                    "Offer Price": offer_details.offer_price,
                    "Deposit": offer_details.deposit,
                    "Conditions": "\n".join(offer_details.conditions),
                    "Closing Date": offer_details.closing_date
                }
                
                completed_pdf = self.pdf_hander.fill_pdf_form(
                    template_name=template_name,
                    output_path=output_path,
                    filed_data=field_data   
                )
                completed_pdfs.append(completed_pdf)
            
            state["completed_pdfs"] = completed_pdfs
            state["offer_details"] = offer_details
            
            return {
                "messages": [
                    SystemMessage(content=f"Completed {len(completed_pdfs)} offer documents.")
                ],
                "completed_pdfs": completed_pdfs,
                "errors": None
            }
        except Exception as e:
            return {
                "messages": [SystemMessage(content=f"An error occurred while preparing the offer document: {str(e)}.")],
                "errors": [str(e)]
            }
    
    def email_documents(self, state: State):
        """Email the completed documents to the buyer and seller."""
        try:
            if not state.get("completed_pdfs"):
                raise ValueError("No completed documents found to email.")
            
            msg = MIMEMultipart()
            msg["Subject"] = f"Offer Documents for Review - {state['offer_details'].buyer_name}"
            msg["From"] = "sender@example.com"
            msg["To"] = "recipient@example.com"
            
            for pdf_path in state["completed_pdfs"]:
                with open(pdf_path, "rb") as f:
                    part = MIMEApplication(f.read(), Name=os.path.basename(pdf_path))
                    part['Content-Disposition'] = f'attachment; filename="{os.path.basename(pdf_path)}"'
                    msg.attach(part)
                    
            # Confiure the SMTP server
            with smtplib.SMTP("smtp.example.com", 587) as server:
                server.starttls()
                server.login("username", "password")
                server.send_message(msg)
                
            return {
                "messages": [
                    SystemMessage(content="Offer documents have been emailed to the buyer and seller.")
                ],
                "errors": None
            }
        except Exception as e:
            return {
                "messages": [SystemMessage(content=f"An error occurred while emailing the offer documents: {str(e)}.")],
                "errors": [str(e)]
            }
    
def setup_workflow():
        """Set up the Langgraph workflow"""
        agent = OfferAgent()
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("prepare_documents", agent.prepare_documents)
        workflow.add_node("email_documents", agent.email_documents)
        
        # Add edges
        workflow.add_edge("prepare_documents", "email_documents")
        workflow.add_edge(START, "email_documents")
        
    
# Usage example
if __name__ == "__main__":
    graph = setup_workflow()
    
    initial_state = {
        "messages": [
            HumanMessage(content="prepare the offer documents for John Smith, "
                        "offer price is $500k, conditions are financing, "
                        "insurance and inspection, deposit is $20k")
        ],
        "offer_details": None,
        "completed_pdfs": [],
        "errors": None
    }
    
    events = graph.stream(initial_state)
    for event in events:
        if "messages" in event:
            for message in event["messages"]:
                print(f"{message.type}: {message.content}")            
                
    
    