dataLayer.push({ submitted_fields: null });  // Clear the previous submitted_fields object

window.dataLayer.push({
  event: "form_start",
  form_id: 'STRING', //Example: form_free_audit | form_free_consult | form_free_contribution | form_free_conversion_guide,
  
  submitted_fields: {
    ck_first_name: "STRING",
    ck_last_name: "STRING",
    ck_email: "STRING",
    ck_company_type: "STRING",
    ck_traffic: "STRING",
    ck_message: "STRING"
  }
});


window.dataLayer.push({
  event: "form_submit",
  form_id: "form_free_audit | form_free_consult | form_free_contribution | form_free_conversion_guide",
  
  submitted_fields: {
    ck_first_name: "Test",
    ck_last_name: "Test",
    ck_email: "test@test.com",
    ck_company_type: "Charity",
    ck_traffic: "50,000 – 100,000",
    ck_message: "Testing message"
  }
});

